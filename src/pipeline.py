import os
import json
import csv
from src.preprocess import preprocess_audio
from src.diarization import diarize_audio
from src.transcribe import transcribe_with_whisperx
import logging
from intervaltree import Interval, IntervalTree

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def merge_segments(segments: list) -> list:
    """
    Merges overlapping and adjacent segments from the same speaker.

    Args:
        segments (list): A list of transcribed segments.

    Returns:
        list: A list of merged segments.
    """
    if not segments:
        return []

    # Sort segments by start time first
    segments.sort(key=lambda x: x['start'])

    # Build an interval tree for each speaker
    speaker_trees = {}
    for i, seg in enumerate(segments):
        speaker = seg.get('speaker', 'UNKNOWN')
        if speaker not in speaker_trees:
            speaker_trees[speaker] = IntervalTree()
        
        # Ensure we have valid start/end times
        start_time = float(seg.get('start', 0))
        end_time = float(seg.get('end', start_time + 0.1))
        
        # Add small buffer to avoid identical intervals
        speaker_trees[speaker].add(Interval(start_time, end_time, (i, seg)))

    # Merge intervals for each speaker
    for speaker in speaker_trees:
        speaker_trees[speaker].merge_overlaps()

    # Reconstruct the merged segments
    merged_segments = []
    for speaker, tree in speaker_trees.items():
        for interval in sorted(tree):
            # Get all original segments that contributed to this merged interval
            original_segments = []
            for iv in tree[interval.begin:interval.end]:
                original_segments.append(iv.data[1])
            
            # Sort by start time
            original_segments.sort(key=lambda x: x['start'])
            
            # Aggregate text and words
            texts = []
            all_words = []
            
            for seg in original_segments:
                if seg.get('text', '').strip():
                    texts.append(seg['text'].strip())
                if seg.get('words'):
                    all_words.extend(seg['words'])
            
            # Create merged segment
            merged_segment = {
                'speaker': speaker,
                'start': interval.begin,
                'end': interval.end,
                'text': ' '.join(texts).strip(),
                'words': all_words
            }
            
            # Only add segments with actual content
            if merged_segment['text']:
                merged_segments.append(merged_segment)

    # Sort the final merged segments by start time
    merged_segments.sort(key=lambda x: x['start'])
    
    logging.info(f"Merged {len(segments)} segments into {len(merged_segments)} final segments")
    return merged_segments

def validate_segments(segments: list) -> list:
    """
    Validates and cleans up segments before final output.
    
    Args:
        segments: List of segments to validate
    
    Returns:
        List of validated segments
    """
    valid_segments = []
    
    for seg in segments:
        # Skip segments without text or with invalid speakers
        if not seg.get('text', '').strip():
            continue
            
        # Clean up speaker labels
        speaker = seg.get('speaker', 'UNKNOWN')
        if speaker in [None, 'None', '']:
            speaker = 'UNKNOWN'
        
        # Ensure proper data types
        try:
            start_time = float(seg.get('start', 0))
            end_time = float(seg.get('end', start_time + 0.1))
        except (ValueError, TypeError):
            logging.warning(f"Invalid timing in segment: {seg}")
            continue
        
        # Skip very short segments
        if end_time - start_time < 0.1:
            continue
        
        valid_segments.append({
            'speaker': str(speaker),
            'start': start_time,
            'end': end_time,
            'text': seg['text'].strip(),
            'words': seg.get('words', [])
        })
    
    logging.info(f"Validated {len(valid_segments)} segments from {len(segments)} original segments")
    return valid_segments

def run_pipeline(input_audio_path: str, output_dir: str, num_speakers: int = 0):
    """
    Runs the full diarization and transcription pipeline.

    Args:
        input_audio_path (str): Path to the user-supplied audio file.
        output_dir (str): Directory to store outputs.
        num_speakers (int): Number of speakers to detect (0 for automatic detection).

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Preprocess
    logging.info("Preprocessing audio...")
    success, processed_audio = preprocess_audio(input_audio_path, output_dir)
    if not success:
        logging.error(f"Preprocessing failed: {processed_audio}")
        return

    # Step 2: Diarization
    logging.info("Running diarization...")
    diarization_segments = diarize_audio(processed_audio, num_speakers)
    if not diarization_segments:
        logging.error("Diarization failed or returned no segments.")
        return

    logging.info(f"Diarization produced {len(diarization_segments)} segments")
    
    # Debug: Print diarization results
    speakers_found = set(seg['speaker'] for seg in diarization_segments)
    logging.info(f"Speakers found in diarization: {sorted(speakers_found)}")

    # Step 3: Format segments for transcription (simplified format)
    formatted_for_transcription = []
    for seg in diarization_segments:
        formatted_for_transcription.append({
            'speaker': seg['speaker'],
            'start': seg['start'],
            'end': seg['end']
        })

    # Step 4: Transcription
    logging.info("Transcribing segments...")
    # Optimize batch processing for available GPU memory
    transcribed_segments_dict = transcribe_with_whisperx(
        processed_audio, 
        formatted_for_transcription, 
        batch_size=32
    )
    
    transcribed_segments = transcribed_segments_dict.get("segments", [])
    if not transcribed_segments:
        logging.error("Transcription failed or returned no results.")
        return

    logging.info(f"Transcription produced {len(transcribed_segments)} segments")
    
    # Debug: Check speaker assignments in transcription
    transcribed_speakers = set(seg.get('speaker', 'None') for seg in transcribed_segments)
    logging.info(f"Speakers found in transcription: {sorted(transcribed_speakers)}")

    # Step 5: Validate segments
    validated_segments = validate_segments(transcribed_segments)
    if not validated_segments:
        logging.error("No valid transcribed segments found.")
        return

    # Step 6: Merge transcribed segments
    logging.info("Merging transcribed segments...")
    merged_results = merge_segments(validated_segments)

    if not merged_results:
        logging.error("No segments remained after merging.")
        return

    # Final speaker count
    final_speakers = set(seg['speaker'] for seg in merged_results)
    logging.info(f"Final output contains {len(merged_results)} segments with speakers: {sorted(final_speakers)}")

    # Step 7: Export results
    base = os.path.splitext(os.path.basename(input_audio_path))[0]
    
    # Export JSON
    json_path = os.path.join(output_dir, f"{base}_diarization.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(merged_results, f, indent=2, ensure_ascii=False)
    
    # Export CSV
    csv_path = os.path.join(output_dir, f"{base}_diarization.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['speaker', 'start', 'end', 'text'])
        writer.writeheader()
        for row in merged_results:
            writer.writerow({
                'speaker': row['speaker'],
                'start': f"{row['start']:.2f}",
                'end': f"{row['end']:.2f}",
                'text': row['text']
            })
    
    # Export TXT (human-readable format)
    txt_path = os.path.join(output_dir, f"{base}_diarization.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        for row in merged_results:
            f.write(f"[{row['start']:.2f}-{row['end']:.2f}] Speaker {row['speaker']}: {row['text']}\n")
    
    logging.info(f"Pipeline complete. Results saved to {output_dir}")
    logging.info(f"Files created:")
    logging.info(f"  - JSON: {json_path}")
    logging.info(f"  - CSV: {csv_path}")
    logging.info(f"  - TXT: {txt_path}")

    # Clean up processed audio file
    if os.path.exists(processed_audio):
        os.remove(processed_audio)
        logging.info(f"Cleaned up intermediate audio file: {processed_audio}")