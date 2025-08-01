"""
utils.py
--------
Helper functions for the multi-speaker diarization project.
"""

from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def post_process_transcript(transcript_segments: List[Dict], speaker_mapping: Dict[str, str]) -> str:
    """
    Post-process transcript by replacing speaker labels with user-provided names or defaults.
    
    Args:
        transcript_segments: List of transcript segments with speaker, start, end, and text
        speaker_mapping: Dictionary mapping speaker labels (e.g., "SPEAKER_00") to user names
    
    Returns:
        Formatted transcript string ready for download
    """
    logging.info("Starting post-processing of transcript...")
    
    if not transcript_segments:
        logging.warning("No transcript segments provided for post-processing")
        return ""
    
    processed_lines = []
    
    for segment in transcript_segments:
        # Extract segment data
        speaker = segment.get('speaker', 'Unknown')
        start_time = segment.get('start', 0)
        end_time = segment.get('end', 0)
        text = segment.get('text', '').strip()
        
        # Replace speaker label with user-provided name or keep default
        # Handle None speaker gracefully
        if speaker is None:
            display_name = "Unknown"
        else:
            display_name = speaker_mapping.get(speaker, speaker)
        
        # Format the line: [start-end] Speaker: text
        formatted_line = f"[{start_time:.2f}s - {end_time:.2f}s] {display_name}: {text}"
        processed_lines.append(formatted_line)
    
    final_transcript = "\n".join(processed_lines)
    logging.info(f"Post-processing complete. Processed {len(processed_lines)} segments.")
    
    return final_transcript

def extract_unique_speakers(transcript_segments: List[Dict]) -> List[str]:
    """
    Extract unique speaker labels from transcript segments.
    
    Args:
        transcript_segments: List of transcript segments
    
    Returns:
        List of unique speaker labels sorted alphabetically
    """
    speakers = set()
    for segment in transcript_segments:
        speaker = segment.get('speaker')
        if speaker:
            speakers.add(speaker)
    
    return sorted(list(speakers))

def create_speaker_mapping(speaker_names: Dict[str, str], unique_speakers: List[str]) -> Dict[str, str]:
    """
    Create a mapping from speaker labels to display names.
    Uses provided names or defaults to original speaker labels.
    
    Args:
        speaker_names: Dictionary of user-provided speaker names
        unique_speakers: List of unique speaker labels from transcript
    
    Returns:
        Complete mapping from speaker labels to display names
    """
    mapping = {}
    
    for speaker in unique_speakers:
        # Use user-provided name if available and not empty, otherwise use original speaker label
        user_name = speaker_names.get(speaker, '')
        mapping[speaker] = user_name if user_name.strip() else speaker
    
    return mapping 