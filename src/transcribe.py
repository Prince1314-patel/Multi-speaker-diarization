import whisperx
import logging
from typing import List, Dict, Optional
import torch
import pandas as pd
import traceback
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def assign_speakers_manually(transcription_segments: List[Dict], diarization_segments: List[Dict]) -> List[Dict]:
    """
    Manual speaker assignment as fallback when WhisperX assignment fails.
    
    Args:
        transcription_segments: List of transcribed segments with timing
        diarization_segments: List of diarization segments with speaker labels
    
    Returns:
        List of segments with assigned speakers
    """
    logging.info("Using manual speaker assignment fallback...")
    
    # Create a mapping of time ranges to speakers
    speaker_map = []
    for diar_seg in diarization_segments:
        speaker_map.append({
            'start': float(diar_seg['start']),
            'end': float(diar_seg['end']),
            'speaker': str(diar_seg['speaker'])
        })
    
    # Sort by start time
    speaker_map.sort(key=lambda x: x['start'])
    
    assigned_segments = []
    
    for trans_seg in transcription_segments:
        trans_start = float(trans_seg.get('start', 0))
        trans_end = float(trans_seg.get('end', trans_start + 1))
        trans_mid = (trans_start + trans_end) / 2
        
        # Find the best matching speaker segment
        best_speaker = "UNKNOWN"
        best_overlap = 0
        
        for spk_seg in speaker_map:
            # Calculate overlap
            overlap_start = max(trans_start, spk_seg['start'])
            overlap_end = min(trans_end, spk_seg['end'])
            overlap_duration = max(0, overlap_end - overlap_start)
            
            if overlap_duration > best_overlap:
                best_overlap = overlap_duration
                best_speaker = spk_seg['speaker']
            
            # Also check if transcript midpoint falls within speaker segment
            if spk_seg['start'] <= trans_mid <= spk_seg['end']:
                if overlap_duration > 0:  # Prefer segments with actual overlap
                    best_speaker = spk_seg['speaker']
                    break
        
        # Create new segment with speaker assignment
        new_segment = trans_seg.copy()
        new_segment['speaker'] = best_speaker
        
        # Ensure words have speaker assignment if they exist
        if 'words' in new_segment and new_segment['words']:
            for word in new_segment['words']:
                if 'speaker' not in word or word['speaker'] is None:
                    word['speaker'] = best_speaker
        
        assigned_segments.append(new_segment)
    
    logging.info(f"Manual assignment complete. Assigned speakers to {len(assigned_segments)} segments.")
    return assigned_segments

def validate_and_fix_diarization(diarization: List[Dict]) -> pd.DataFrame:
    """
    Validates and fixes diarization data format for WhisperX compatibility.
    
    Args:
        diarization: List of diarization segments
    
    Returns:
        Properly formatted pandas DataFrame
    """
    if not diarization:
        logging.warning("Empty diarization data provided")
        return pd.DataFrame(columns=['speaker', 'start', 'end'])
    
    # Convert to DataFrame and validate
    df = pd.DataFrame(diarization)
    
    required_cols = ['speaker', 'start', 'end']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Diarization data missing required columns: {missing_cols}")
    
    # Ensure proper data types
    df['start'] = pd.to_numeric(df['start'], errors='coerce')
    df['end'] = pd.to_numeric(df['end'], errors='coerce')
    df['speaker'] = df['speaker'].astype(str)
    
    # Remove invalid entries
    before_count = len(df)
    df = df.dropna(subset=['start', 'end'])
    df = df[df['start'] < df['end']]  # Remove invalid time ranges
    after_count = len(df)
    
    if before_count != after_count:
        logging.warning(f"Removed {before_count - after_count} invalid diarization segments")
    
    # Sort by start time
    df = df.sort_values('start').reset_index(drop=True)
    
    logging.info(f"Validated diarization: {len(df)} segments, speakers: {df['speaker'].unique().tolist()}")
    return df

def transcribe_with_whisperx(
    audio_path: str,
    diarization: List[Dict],
    model_name: str = "large-v2",
    device: str = "cuda",
    batch_size: int = 16,
    compute_type: str = "float16"
) -> Dict:
    """
    Transcribes an audio file using WhisperX and aligns with diarization.

    Args:
        audio_path (str): Path to the audio file.
        diarization (List[Dict]): List of diarization segments with 'speaker', 'start', and 'end'.
        model_name (str): Name of the WhisperX model to use.
        device (str): Device to run the model on (e.g., "cuda", "cpu").
        batch_size (int): Batch size for transcription.
        compute_type (str): Compute type for the model (e.g., "float16", "int8").

    Returns:
        Dict: A dictionary containing a list of transcribed segments under the 'segments' key.
    """
    try:
        # 1. Load the WhisperX model
        logging.info(f"Loading whisperx model: {model_name} on {device}")
        model = whisperx.load_model(model_name, device, compute_type=compute_type)

        # 2. Load the audio
        logging.info("Loading audio file...")
        audio = whisperx.load_audio(audio_path)

        # 3. Transcribe the audio
        logging.info("Transcribing audio with WhisperX...")
        result = model.transcribe(audio, batch_size=batch_size)
        
        # Check if transcription produced any segments
        if not result.get("segments"):
            logging.warning("Transcription produced no segments.")
            return {"segments": []}

        logging.info(f"Transcription complete. Found {len(result['segments'])} segments in language: {result.get('language', 'unknown')}")

        # 4. Align the transcription
        logging.info("Loading alignment model...")
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        
        logging.info("Aligning transcription...")
        aligned_result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

        # Filter out segments that have no words after alignment
        segments_with_words = [
            segment for segment in aligned_result["segments"] 
            if 'words' in segment and len(segment['words']) > 0
        ]
        
        if not segments_with_words:
            logging.warning("No segments with words found after alignment.")
            return {"segments": []}
        
        aligned_result["segments"] = segments_with_words
        logging.info(f"Alignment complete. {len(segments_with_words)} segments with words.")

        # 5. Validate and prepare diarization data
        if not diarization:
            logging.warning("No diarization data provided. Cannot assign speakers.")
            return aligned_result

        try:
            diarize_df = validate_and_fix_diarization(diarization)
            if diarize_df.empty:
                logging.warning("Diarization DataFrame is empty after validation.")
                return aligned_result

            # 6. Attempt WhisperX speaker assignment
            logging.info("Attempting WhisperX speaker assignment...")
            
            # Debug: Print sample data
            logging.info(f"Sample diarization data:\n{diarize_df.head()}")
            logging.info(f"Sample transcription segment: {aligned_result['segments'][0] if aligned_result['segments'] else 'None'}")
            
            final_result = whisperx.assign_word_speakers(diarize_df, aligned_result)
            
            # Check if assignment was successful
            successful_assignments = 0
            total_segments = len(final_result.get("segments", []))
            
            for segment in final_result.get("segments", []):
                if segment.get('speaker') and segment['speaker'] != 'UNKNOWN':
                    successful_assignments += 1
            
            success_rate = successful_assignments / total_segments if total_segments > 0 else 0
            logging.info(f"WhisperX assignment success rate: {success_rate:.2%} ({successful_assignments}/{total_segments})")
            
            # If WhisperX assignment failed, use manual assignment
            if success_rate < 0.5:  # Less than 50% success rate
                logging.warning("WhisperX speaker assignment had low success rate. Using manual assignment.")
                final_result["segments"] = assign_speakers_manually(aligned_result["segments"], diarization)
            
        except Exception as e:
            logging.error(f"Error in WhisperX speaker assignment: {e}")
            logging.info("Falling back to manual speaker assignment...")
            final_result = aligned_result.copy()
            final_result["segments"] = assign_speakers_manually(aligned_result["segments"], diarization)

        # 7. Clean up and finalize
        if "segments" in final_result and final_result["segments"]:
            for segment in final_result["segments"]:
                if 'text' in segment:
                    segment['text'] = segment['text'].strip()
                
                # Ensure speaker is properly assigned
                if not segment.get('speaker') or segment['speaker'] in [None, 'None', '']:
                    segment['speaker'] = 'UNKNOWN'

        # Final validation
        total_segments = len(final_result.get("segments", []))
        assigned_segments = sum(1 for seg in final_result.get("segments", []) 
                              if seg.get('speaker') and seg['speaker'] not in ['None', None, 'UNKNOWN'])
        
        logging.info(f"Final result: {total_segments} segments, {assigned_segments} with speaker assignments")
        
        return final_result

    except Exception as e:
        logging.error(f"Error during WhisperX transcription: {e}")
        logging.error(traceback.format_exc())
        return {"segments": []}