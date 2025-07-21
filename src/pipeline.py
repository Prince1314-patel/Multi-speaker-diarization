import os
import json
import csv
from src.preprocess import preprocess_audio
from src.diarization import diarize_audio
from src.transcribe import transcribe_segments
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

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
    # Step 3: Format segments
    formatted_segments = [
        {'speaker': seg[0], 'start': float(seg[1]), 'end': float(seg[2])}
        for seg in diarization_segments
    ]
    # Step 4: Transcription
    logging.info("Transcribing segments...")
    transcribed_segments = transcribe_segments(processed_audio, formatted_segments, temp_dir=output_dir)
    if not transcribed_segments:
        logging.error("Transcription failed or returned no results.")
        return
    # Step 5: Aggregate results
    results = []
    for seg in transcribed_segments:
        results.append({
            'speaker': seg.get('speaker'),
            'start': seg.get('start'),
            'end': seg.get('end'),
            'text': seg.get('text', ''),
            'words': seg.get('words', [])
        })
    # Step 6: Export
    base = os.path.splitext(os.path.basename(input_audio_path))[0]
    # Export JSON
    json_path = os.path.join(output_dir, f"{base}_diarization.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    # Export CSV
    csv_path = os.path.join(output_dir, f"{base}_diarization.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['speaker', 'start', 'end', 'text'])
        writer.writeheader()
        for row in results:
            writer.writerow({
                'speaker': row['speaker'],
                'start': row['start'],
                'end': row['end'],
                'text': row['text']
            })
    # Export TXT
    txt_path = os.path.join(output_dir, f"{base}_diarization.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        for row in results:
            f.write(f"[{row['start']:.2f}-{row['end']:.2f}] Speaker {row['speaker']}: {row['text']}\n")
    logging.info(f"Pipeline complete. Results saved to {output_dir}") 