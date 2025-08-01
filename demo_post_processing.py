#!/usr/bin/env python3
"""
Demo script for post-processing functionality.
This script demonstrates how the speaker name assignment and transcript formatting works.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils import post_process_transcript, extract_unique_speakers, create_speaker_mapping

def demo_post_processing():
    """Demonstrate the post-processing functionality."""
    
    print("=== Multi-Speaker Diarization Post-Processing Demo ===\n")
    
    # Mock transcript segments (similar to what would come from transcription)
    transcript_segments = [
        {'speaker': 'SPEAKER_00', 'start': 0.0, 'end': 3.5, 'text': 'Hello, how are you today?'},
        {'speaker': 'SPEAKER_01', 'start': 3.5, 'end': 6.2, 'text': 'I\'m doing well, thank you.'},
        {'speaker': 'SPEAKER_00', 'start': 6.2, 'end': 9.8, 'text': 'That\'s great to hear.'},
        {'speaker': 'SPEAKER_01', 'start': 9.8, 'end': 12.5, 'text': 'How about yourself?'},
        {'speaker': 'SPEAKER_00', 'start': 12.5, 'end': 15.0, 'text': 'Pretty good, thanks for asking.'}
    ]
    
    print("Original transcript segments:")
    for segment in transcript_segments:
        print(f"  {segment['speaker']}: {segment['text']}")
    
    print("\n" + "="*50)
    
    # Extract unique speakers
    unique_speakers = extract_unique_speakers(transcript_segments)
    print(f"\nDetected speakers: {unique_speakers}")
    
    # Simulate user input for speaker names
    print("\nUser assigns names to speakers:")
    speaker_names = {
        'SPEAKER_00': 'Prince',  # User provided name
        'SPEAKER_01': '',        # User left empty (will use default)
    }
    
    for speaker, name in speaker_names.items():
        if name:
            print(f"  {speaker} -> {name}")
        else:
            print(f"  {speaker} -> (using default)")
    
    # Create speaker mapping
    speaker_mapping = create_speaker_mapping(speaker_names, unique_speakers)
    print(f"\nFinal speaker mapping: {speaker_mapping}")
    
    # Post-process transcript
    processed_transcript = post_process_transcript(transcript_segments, speaker_mapping)
    
    print("\n" + "="*50)
    print("FINAL PROCESSED TRANSCRIPT:")
    print("="*50)
    print(processed_transcript)
    print("="*50)
    
    # Show what the downloaded file would look like
    print("\nThis transcript would be saved as 'transcript.txt' with the following content:")
    print("-" * 50)
    print(processed_transcript)
    print("-" * 50)

if __name__ == "__main__":
    demo_post_processing() 