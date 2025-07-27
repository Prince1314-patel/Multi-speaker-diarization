
import os
import glob
from src.pipeline import run_pipeline

# Create the output directory if it doesn't exist
output_dir = "transcripts"
os.makedirs(output_dir, exist_ok=True)

# Find all audio files in the input directory
audio_files = glob.glob("audio_files/*")

# Process each audio file
for audio_file in audio_files:
    print(f"Processing {audio_file}...")
    run_pipeline(audio_file, output_dir)
    print(f"Finished processing {audio_file}")
