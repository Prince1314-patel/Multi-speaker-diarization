# diarization.py

from typing import List, Dict
import os
import logging
from pyannote.audio import Pipeline as DiarizationPipeline
from pyannote.core import Annotation, Segment, Timeline
import torch
from src.overlap import OverlapDetector
from torch.cuda import empty_cache

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def resolve_overlaps(diarization: Annotation, overlap_regions: List[tuple]) -> List[Dict[str, any]]:
    """
    Resolves overlaps in a diarization by creating multi-speaker segments.
    """
    original_timeline = diarization.get_timeline()
    resolved_segments = []

    if overlap_regions:
        overlap_timeline = Timeline([Segment(start, end) for start, end in overlap_regions])
        non_overlap_timeline = original_timeline.gaps(overlap_timeline)
    else:
        non_overlap_timeline = original_timeline

    # Process non-overlap segments
    for segment in non_overlap_timeline:
        speakers = list(diarization.crop(segment).labels())
        if speakers:
            resolved_segments.append({
                'speaker': str(speakers[0]),  # Ensure speaker is string
                'start': float(segment.start),  # Ensure numeric type
                'end': float(segment.end),  # Ensure numeric type
                'is_overlap': False,
                'overlap_with': []
            })
           
    # Process overlap segments
    for start, end in overlap_regions:
        overlap_segment = Segment(start, end)
        speakers_in_overlap = list(diarization.crop(overlap_segment).labels())
       
        if len(speakers_in_overlap) > 1:
            for speaker in speakers_in_overlap:
                resolved_segments.append({
                    'speaker': str(speaker),  # Ensure speaker is string
                    'start': float(start),  # Ensure numeric type
                    'end': float(end),  # Ensure numeric type
                    'is_overlap': True,
                    'overlap_with': [str(s) for s in speakers_in_overlap if s != speaker]
                })

    # Sort by start time and ensure no duplicate or invalid segments
    resolved_segments.sort(key=lambda x: x['start'])
    
    # Filter out very short segments (< 0.1 seconds) that might cause issues
    resolved_segments = [seg for seg in resolved_segments if seg['end'] - seg['start'] >= 0.1]
    
    return resolved_segments

def format_diarization_for_whisperx(diarization: Annotation) -> List[Dict[str, any]]:
    """
    Formats pyannote diarization output for WhisperX compatibility.
    
    Args:
        diarization: Pyannote Annotation object
    
    Returns:
        List of segments formatted for WhisperX
    """
    segments = []
    
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        # Ensure proper data types and format
        segments.append({
            'speaker': str(speaker),  # Convert to string
            'start': float(turn.start),  # Ensure float
            'end': float(turn.end),  # Ensure float
            'is_overlap': False,
            'overlap_with': []
        })
    
    # Sort by start time
    segments.sort(key=lambda x: x['start'])
    
    # Merge very close segments from the same speaker (< 0.1 second gap)
    merged_segments = []
    for segment in segments:
        if (merged_segments and 
            merged_segments[-1]['speaker'] == segment['speaker'] and
            segment['start'] - merged_segments[-1]['end'] < 0.1):
            # Merge with previous segment
            merged_segments[-1]['end'] = segment['end']
        else:
            merged_segments.append(segment)
    
    # Filter out very short segments
    merged_segments = [seg for seg in merged_segments if seg['end'] - seg['start'] >= 0.1]
    
    logging.info(f"Formatted {len(merged_segments)} segments for WhisperX")
    return merged_segments

def diarize_audio(audio_path: str, num_speakers: int = 0) -> List[Dict[str, any]]:
    """
    Performs speaker diarization with active overlap resolution.

    Args:
        audio_path (str): Path to the preprocessed WAV audio file.
        num_speakers (int): The number of speakers to detect. If 0, the model detects automatically.

    Returns:
        List[Dict[str, any]]: List of segments with resolved overlaps.
    """
    try:
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        if not hf_token:
            logging.error("Hugging Face token not found in environment variable 'HUGGINGFACE_TOKEN'.")
            return []
       
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")

        # 1. Run standard diarization
        logging.info("Loading pyannote/speaker-diarization-3.1 pipeline...")
        diarization_pipeline = DiarizationPipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        ).to(device)

        # Use the num_speakers parameter if provided by the user
        pipeline_params = {}
        if num_speakers > 0:
            pipeline_params["num_speakers"] = num_speakers
            logging.info(f"Diarizing with a fixed number of speakers: {num_speakers}")
        else:
            logging.info("Diarizing with automatic speaker number detection.")
       
        diarization = diarization_pipeline(audio_path, **pipeline_params)
        logging.info("Initial diarization complete.")

        # Log detected speakers
        speakers = list(diarization.labels())
        logging.info(f"Detected speakers: {speakers}")

        if torch.cuda.is_available():
            empty_cache()

        # 2. Detect overlap regions
        logging.info("Loading pyannote/overlapped-speech-detection pipeline...")
        overlap_detector = OverlapDetector(use_cuda=torch.cuda.is_available())
        if not overlap_detector.initialize(auth_token=hf_token):
            logging.warning("Overlap detector failed to initialize. Using standard diarization.")
            final_segments = format_diarization_for_whisperx(diarization)
            return final_segments

        overlap_regions = overlap_detector.detect_overlaps(audio_path)
        logging.info(f"Detected {len(overlap_regions)} overlap regions.")
       
        # 3. Resolve overlaps and create the final segment list
        if overlap_regions:
            final_segments = resolve_overlaps(diarization, overlap_regions)
        else:
            logging.info("No overlaps detected. Using standard diarization format.")
            final_segments = format_diarization_for_whisperx(diarization)

        # 4. Final validation and logging
        num_overlaps = sum(1 for seg in final_segments if seg.get('is_overlap'))
        unique_speakers = set(seg['speaker'] for seg in final_segments)
        
        logging.info(f"Diarization complete:")
        logging.info(f"  - Total segments: {len(final_segments)}")
        logging.info(f"  - Overlap segments: {num_overlaps}")
        logging.info(f"  - Unique speakers: {sorted(unique_speakers)}")

        # Debug: Print first few segments
        if final_segments:
            logging.info("Sample segments:")
            for i, seg in enumerate(final_segments[:3]):
                logging.info(f"  Segment {i+1}: Speaker {seg['speaker']}, {seg['start']:.2f}-{seg['end']:.2f}s")

        return final_segments

    except Exception as e:
        logging.error(f"Error during diarization: {e}", exc_info=True)
        return []