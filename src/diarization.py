from typing import List, Tuple
import os
import logging
from pyannote.audio import Pipeline
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def diarize_audio(audio_path: str, window_size: float = 5.0, step_size: float = 0.5) -> List[Tuple[str, float, float, bool]]:
    """
    Performs speaker diarization on the given audio file using pyannote.audio.
    Uses CUDA if available, otherwise CPU. Implements overlapping window processing
    for better handling of overlapped speech.

    Args:
        audio_path (str): Path to the preprocessed WAV audio file.
        window_size (float): Size of the processing window in seconds (default: 5.0)
        step_size (float): Step size between windows in seconds (default: 0.5)

    Returns:
        List[Tuple[str, float, float, bool]]: List of (speaker_label, start_time, end_time, is_overlap) segments.
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
        
        # Create a timeline of all segments to detect overlaps
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # Check for overlapping segments
            overlapping = False
            current_segment = timeline.crop(turn.start, turn.end)
            
            # If the segment overlaps with any other segment
            if len(current_segment) > 1:
                overlapping = True
                logging.info(f"Detected overlap at {turn.start:.2f}-{turn.end:.2f}")
            
            segments.append((speaker, turn.start, turn.end, overlapping))
            
        # Sort segments by start time
        segments.sort(key=lambda x: x[1])
        
        logging.info(f"Diarization complete. Found {len(segments)} segments, including overlaps.")
        return segments
    except Exception as e:
        logging.error(f"Error during diarization: {e}")
        return [] 