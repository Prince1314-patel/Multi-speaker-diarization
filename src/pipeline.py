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

    # Build an interval tree for each speaker
    speaker_trees = {}
    for i, seg in enumerate(segments):
        speaker = seg.get('speaker')
        if speaker not in speaker_trees:
            speaker_trees[speaker] = IntervalTree()
        # Use a unique ID in the data to handle identical intervals
        speaker_trees[speaker].add(Interval(seg['start'], seg['end'], (i, seg)))

    # Merge intervals for each speaker
    for speaker in speaker_trees:
        speaker_trees[speaker].merge_overlaps()

    # Reconstruct the merged segments
    merged_segments = []
    for speaker, tree in speaker_trees.items():
        for interval in sorted(tree):
            # Aggregate text and words from original segments
            original_indices = {iv.data[0] for iv in tree[interval.begin:interval.end]}
            text = " ".join([segments[i]['text'] for i in sorted(original_indices)])
            words = sum([segments[i]['words'] for i in sorted(original_indices)], [])
            
            merged_segments.append({
                'speaker': speaker,
                'start': interval.begin,
                'end': interval.end,
                'text': text.strip(),
                'words': words
            })

    # Sort the final merged segments by start time
    merged_segments.sort(key=lambda x: x['start'])
    return merged_segments

def run_pipeline(input_audio_path: str, output_dir: str):
    """
    Runs the full diarization and transcription pipeline.

    Args:
        input_audio_path (str): Path to the user-supplied audio file.
        output_dir (str): Directory to store outputs.

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
    diarization_segments = diarize_audio(processed_audio)
    if not diarization_segments:
        logging.error("Diarization failed or returned no segments.")
        return

    # Step 3: Format segments for transcription
    formatted_for_transcription = []
    for seg in diarization_segments:
        formatted_for_transcription.append({
            'speaker': seg['speaker'],
            'start': seg['start'],
            'end': seg['end']
        })

    # Step 4: Transcription
    logging.info("Transcribing segments...")
    # Step 4: Transcription
    logging.info("Transcribing segments...")
    # Optimize batch processing for L4 GPU (24GB VRAM)
    # A batch_size of 32 is a common starting point for L4 GPUs, adjust as needed.
    transcribed_segments_dict = transcribe_with_whisperx(processed_audio, formatted_for_transcription, batch_size=32)
    transcribed_segments = transcribed_segments_dict.get("segments", [])
    if not transcribed_segments:
        logging.error("Transcription failed or returned no results.")
        return

    # Step 5: Merge transcribed segments
    logging.info("Merging transcribed segments...")
    merged_results = merge_segments(transcribed_segments)

    # Step 6: Export
    base = os.path.splitext(os.path.basename(input_audio_path))[0]
    # Export JSON
    json_path = os.path.join(output_dir, f"{base}_diarization.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(merged_results, f, indent=2)
    # Export CSV
    csv_path = os.path.join(output_dir, f"{base}_diarization.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['speaker', 'start', 'end', 'text'])
        writer.writeheader()
        for row in merged_results:
            writer.writerow({
                'speaker': row['speaker'],
                'start': row['start'],
                'end': row['end'],
                'text': row['text']
            })
    # Export TXT
    txt_path = os.path.join(output_dir, f"{base}_diarization.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        for row in merged_results:
            f.write(f"[{row['start']:.2f}-{row['end']:.2f}] Speaker {row['speaker']}: {row['text']}\n")
    logging.info(f"Pipeline complete. Results saved to {output_dir}")

    # Clean up processed audio file
    if os.path.exists(processed_audio):
        os.remove(processed_audio)
        logging.info(f"Cleaned up intermediate audio file: {processed_audio}") 