from typing import List, Tuple, Dict
import os
import logging
import numpy as np
from pyannote.audio import Pipeline
import torch
from src.overlap import analyze_overlap, merge_overlapping_segments
from torch.cuda import empty_cache

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def diarize_audio(audio_path: str, window_size: float = 5.0, step_size: float = 0.5, 
                min_overlap_duration: float = 0.2) -> List[Dict[str, any]]:
    """
    Performs speaker diarization on the given audio file using pyannote.audio.
    Uses CUDA if available, otherwise CPU. Implements advanced overlapping window processing
    with enhanced overlap detection and segment merging.

    Args:
        audio_path (str): Path to the preprocessed WAV audio file.
        window_size (float): Size of the processing window in seconds (default: 5.0)
        step_size (float): Step size between windows in seconds (default: 0.5)
        min_overlap_duration (float): Minimum duration to consider as overlap (default: 0.2)

    Returns:
        List[Dict[str, any]]: List of segments with following structure:
            {
                'speaker': str,           # Speaker label
                'start': float,          # Start time in seconds
                'end': float,            # End time in seconds
                'is_overlap': bool,      # Whether segment overlaps with others
                'overlap_with': List[str] # List of speaker labels this segment overlaps with
            }
    """
    try:
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        if not hf_token:
            logging.error("Hugging Face token not found in environment variable 'HUGGINGFACE_TOKEN'.")
            return []
        logging.info("Loading pyannote.audio pipeline...")
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
        if torch.cuda.is_available():
            device = torch.device("cuda")
            pipeline.to(device)
            logging.info("Using CUDA for diarization.")
        else:
            device = torch.device("cpu")
            pipeline.to(device)
            logging.info("Using CPU for diarization.")
        logging.info(f"Processing audio for diarization: {audio_path}")
        
        # Apply diarization with overlapping windows
        diarization = pipeline(audio_path, segmentation={"window": window_size, "step": step_size})
        
        # Process segments and detect overlaps
        segments = []
        timeline = diarization.get_timeline()
        
        # First pass: collect all segments
        raw_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            raw_segments.append({
                'speaker': speaker,
                'start': turn.start,
                'end': turn.end,
                'is_overlap': False,
                'overlap_with': []
            })
            
        # Clear CUDA cache after initial processing
        if torch.cuda.is_available():
            empty_cache()
            
        # Analyze overlaps using the overlap detection module
        processed_segments = analyze_overlap(
            raw_segments,
            min_duration=min_overlap_duration
        )
        
        # Merge overlapping segments that belong to the same speaker
        final_segments = merge_overlapping_segments(processed_segments)
        
        # Sort segments by start time
        final_segments.sort(key=lambda x: x['start'])
        
        num_overlaps = sum(1 for seg in final_segments if seg['is_overlap'])
        logging.info(f"Diarization complete. Found {len(final_segments)} segments, "
                    f"including {num_overlaps} overlapping segments.")
        
        return final_segments
    except Exception as e:
        logging.error(f"Error during diarization: {e}")
        return [] 