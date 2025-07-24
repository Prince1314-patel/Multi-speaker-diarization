from typing import List, Tuple, Dict
import os
import logging
import numpy as np
from pyannote.audio import Pipeline
import torch
from src.overlap import OverlapDetector
from torch.cuda import empty_cache

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def diarize_audio(audio_path: str, min_overlap_duration: float = 0.2) -> List[Dict[str, any]]:
    """
    Performs speaker diarization on the given audio file using pyannote.audio.
    Uses CUDA if available, otherwise CPU. Implements advanced overlapping window processing
    with enhanced overlap detection and segment merging.

    Args:
        audio_path (str): Path to the preprocessed WAV audio file.
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
        
        diarization = pipeline(audio_path)
        
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
        overlap_detector = OverlapDetector()
        if overlap_detector.initialize(hf_token):
            overlap_regions = overlap_detector.detect_overlaps(audio_path)
            # Merge diarization with overlaps
            enhanced_segments = overlap_detector.merge_with_diarization(
                [(seg['speaker'], seg['start'], seg['end']) for seg in raw_segments],
                overlap_regions
            )
            # Convert back to list of dicts
            processed_segments = [
                {
                    'speaker': speaker,
                    'start': start,
                    'end': end,
                    'is_overlap': is_overlap,
                    'overlap_with': []  # This part needs to be implemented if needed
                }
                for speaker, start, end, is_overlap in enhanced_segments
            ]
        else:
            processed_segments = raw_segments
        
        final_segments = processed_segments
        
        # Sort segments by start time
        final_segments.sort(key=lambda x: x['start'])
        
        num_overlaps = sum(1 for seg in final_segments if seg['is_overlap'])
        logging.info(f"Diarization complete. Found {len(final_segments)} segments, "
                    f"including {num_overlaps} overlapping segments.")
        
        return final_segments
    except Exception as e:
        logging.error(f"Error during diarization: {e}")
        return [] 