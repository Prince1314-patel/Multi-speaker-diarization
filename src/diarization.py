from typing import List, Tuple
import os
import logging
from pyannote.audio import Pipeline
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def diarize_audio(audio_path: str) -> List[Tuple[str, float, float]]:
    """
    Performs speaker diarization on the given audio file using pyannote.audio.
    Uses CUDA if available, otherwise CPU.

    Args:
        audio_path (str): Path to the preprocessed WAV audio file.

    Returns:
        List[Tuple[str, float, float]]: List of (speaker_label, start_time, end_time) segments.
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
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append((speaker, turn.start, turn.end))
        logging.info(f"Diarization complete. Found {len(segments)} segments.")
        return segments
    except Exception as e:
        logging.error(f"Error during diarization: {e}")
        return [] 