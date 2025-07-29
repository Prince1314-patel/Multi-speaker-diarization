import os
import json
import csv
import logging
from typing import List, Dict, Any, Tuple
from intervaltree import Interval, IntervalTree

from src.preprocess import preprocess_audio
from src.diarization import diarize_audio
from src.transcribe import transcribe_with_whisperx

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def merge_segments_reducer(a, b):
    """
    A reducer function for intervaltree to correctly merge segment data.
    It combines the 'text' and 'words' from two segments.
    """
    # a is the accumulated data, b is the data from the new interval being merged.
    # The data is a tuple: (original_index, segment_dict)
    
    # Unpack the data
    _idx_a, seg_a = a
    _idx_b, seg_b = b

    # Create a new combined segment dictionary
    combined_seg = {
        'text': (seg_a.get('text', '') + ' ' + seg_b.get('text', '')).strip(),
        'words': seg_a.get('words', []) + seg_b.get('words', []),
        # Keep other relevant fields if necessary, like speaker
        'speaker': seg_a.get('speaker') 
    }
    
    # The new data for the merged interval is a placeholder index (-1) and the combined dict
    return (-1, combined_seg)

def merge_segments(segments: List) -> List:
    """
    Merges overlapping and adjacent transcribed segments from the same speaker
    using a robust reduction method.

    Args:
        segments (list): A list of transcribed segments from whisperx.

    Returns:
        list: A list of merged segments.
    """
    if not segments:
        return []

    speaker_trees = {}
    for i, seg in enumerate(segments):
        speaker = seg.get('speaker')
        if speaker is None: # Handle unassigned speakers gracefully
            speaker = "UNKNOWN_SPEAKER"
        
        if speaker not in speaker_trees:
            speaker_trees[speaker] = IntervalTree()
        
        # Store the original index and the full segment data in the interval
        speaker_trees[speaker].add(Interval(seg['start'], seg['end'], (i, seg)))

    # Merge intervals for each speaker using the safe reducer
    for speaker in speaker_trees:
        # The reducer ensures that data is aggregated correctly only from
        # the intervals that are actually being merged.
        speaker_trees[speaker].merge_overlaps(data_reducer=merge_segments_reducer)

    # Reconstruct the final list of segments from the merged trees
    merged_segments = []
    for speaker, tree in speaker_trees.items():
        for interval in sorted(tree):
            _idx, data = interval.data
            merged_segments.append({
                'speaker': speaker if speaker != "UNKNOWN_SPEAKER" else None,
                'start': interval.begin,
                'end': interval.end,
                'text': data.get('text', '').strip(),
                'words': data.get('words', [])
            })

    merged_segments.sort(key=lambda x: x['start'])
    return merged_segments

def enrich_with_overlaps(segments: List, overlap_regions: List) -> List:
    """
    Enriches transcribed segments with overlap information.

    This is the final "Enrich" step. It iterates through the final segments and
    adds an 'is_overlap' flag if the segment's time range intersects with a
    detected overlap region.

    Args:
        segments (List): The final list of transcribed and merged segments.
        overlap_regions (List[Tuple[float, float]]): List of (start, end) tuples for overlaps.

    Returns:
        List: The list of segments, with each segment now containing an 'is_overlap' key.
    """
    if not overlap_regions:
        for seg in segments:
            seg['is_overlap'] = False
        return segments

    overlap_tree = IntervalTree()
    for start, end in overlap_regions:
        overlap_tree.add(Interval(start, end))

    for seg in segments:
        # Check if the segment's interval [start, end] overlaps with any interval in the tree.
        seg['is_overlap'] = overlap_tree.overlaps(seg['start'], seg['end'])
        
    return segments

def run_pipeline(input_audio_path: str, output_dir: str, num_speakers: int = 0):
    """
    Runs the full, refactored diarization and transcription pipeline.

    Args:
        input_audio_path (str): Path to the user-supplied audio file.
        output_dir (str): Directory to store outputs.
        num_speakers (int): The number of speakers to detect (0 for automatic).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Preprocess
    logging.info("Preprocessing audio...")
    success, processed_audio = preprocess_audio(input_audio_path, output_dir)
    if not success:
        logging.error(f"Preprocessing failed: {processed_audio}") # <-- THIS IS THE FIX
        return

    # Step 2: Diarize (New "Diarize" step)
    # This now returns a clean timeline AND separate overlap regions.
    logging.info("Running diarization and overlap detection...")
    diarization_segments, overlap_regions = diarize_audio(processed_audio, num_speakers)
    if not diarization_segments:
        logging.error("Diarization failed or returned no segments.")
        if os.path.exists(processed_audio):
            os.remove(processed_audio)
        return

    # Step 3: Transcription and Alignment (New "Align" step)
    # This step now receives the clean, unambiguous timeline and will succeed.
    logging.info("Transcribing segments with WhisperX...")
    transcribed_result = transcribe_with_whisperx(
        processed_audio, 
        diarization_segments, 
        batch_size=32
    )
    transcribed_segments = transcribed_result.get("segments", [])
    if not transcribed_segments:
        logging.error("Transcription failed or returned no results.")
        if os.path.exists(processed_audio):
            os.remove(processed_audio)
        return

    # Step 4: Merge transcribed segments
    logging.info("Merging adjacent transcribed segments...")
    merged_segments = merge_segments(transcribed_segments)

    # Step 5: Enrich with Overlap Data (New "Enrich" step)
    # This is the final post-processing step to add overlap metadata.
    logging.info("Enriching transcript with overlap information...")
    final_results = enrich_with_overlaps(merged_segments, overlap_regions)

    # Step 6: Export results
    base = os.path.splitext(os.path.basename(input_audio_path))[0]
    logging.info("Exporting final results...")

    # Export JSON (with new 'is_overlap' field)
    json_path = os.path.join(output_dir, f"{base}_transcript.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2)

    # Export CSV
    csv_path = os.path.join(output_dir, f"{base}_transcript.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        # Add 'is_overlap' to the header
        writer = csv.DictWriter(f, fieldnames=['speaker', 'start', 'end', 'text', 'is_overlap'])
        writer.writeheader()
        for row in final_results:
            writer.writerow({
                'speaker': row.get('speaker'),
                'start': row.get('start'),
                'end': row.get('end'),
                'text': row.get('text'),
                'is_overlap': row.get('is_overlap', False)
            })

    # Export TXT
    txt_path = os.path.join(output_dir, f"{base}_transcript.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        for row in final_results:
            overlap_marker = " (OVERLAP)" if row.get('is_overlap') else ""
            speaker_label = row.get('speaker', 'UNKNOWN')
            f.write(f"[{row['start']:.2f}-{row['end']:.2f}] {speaker_label}{overlap_marker}: {row['text']}\n")

    logging.info(f"Pipeline complete. Results saved to {output_dir}")

    # Clean up intermediate processed audio file
    if os.path.exists(processed_audio):
        os.remove(processed_audio)
        logging.info(f"Cleaned up intermediate audio file: {processed_audio}")