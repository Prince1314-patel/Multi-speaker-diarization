import os
import logging
from typing import List, Dict, Any, Tuple
import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from torch.cuda import empty_cache

load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def diarize_audio(audio_path: str, num_speakers: int = 0) -> Tuple[List[Dict[str, Any]], List[Tuple[float, float]]]:
    """
    Performs speaker diarization and overlap detection as two separate steps.

    This function first runs a standard diarization pipeline to get a clean,
    non-overlapping timeline. Then, it runs a separate overlap detection
    pipeline. It returns both results without trying to merge them.

    Args:
        audio_path (str): Path to the preprocessed WAV audio file.
        num_speakers (int): The number of speakers to detect. If 0, detects automatically.

    Returns:
        A tuple containing:
        - List[Dict[str, Any]]: A clean list of diarization segments for transcription.
        - List[Tuple[float, float]]: A list of (start, end) tuples for detected overlap regions.
    """
    diarization_segments = []
    overlap_regions = []
    
    try:
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        if not hf_token:
            logging.error("Hugging Face token not found. Set 'HUGGINGFACE_TOKEN' env variable.")
            return [], []

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")

        # 1. Standard Diarization Pipeline
        logging.info("Loading pyannote/speaker-diarization-3.1 pipeline...")
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        ).to(device)

        pipeline_params = {}
        if num_speakers > 0:
            pipeline_params["num_speakers"] = num_speakers
            logging.info(f"Diarizing with fixed number of speakers: {num_speakers}")
        else:
            logging.info("Diarizing with automatic speaker number detection.")

        diarization = diarization_pipeline(audio_path, **pipeline_params)
        
        # Format diarization for whisperx
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            diarization_segments.append({
                'speaker': speaker,
                'start': turn.start,
                'end': turn.end
            })
        logging.info(f"Initial diarization complete. Found {len(diarization_segments)} segments.")

        if torch.cuda.is_available():
            empty_cache()

        # 2. Overlap Detection Pipeline
        logging.info("Loading pyannote/overlapped-speech-detection pipeline...")
        overlap_pipeline = Pipeline.from_pretrained(
            "pyannote/overlapped-speech-detection",
            use_auth_token=hf_token
        ).to(device)

        overlaps = overlap_pipeline(audio_path)
        
        # Format overlap regions
        for segment in overlaps.get_timeline():
            overlap_regions.append((segment.start, segment.end))
        logging.info(f"Overlap detection complete. Found {len(overlap_regions)} overlap regions.")

        return diarization_segments, overlap_regions

    except Exception as e:
        logging.error(f"Error during diarization or overlap detection: {e}", exc_info=True)
        # Return empty lists in case of any failure
        return [], []