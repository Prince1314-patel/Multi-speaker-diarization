from typing import List, Tuple
import os
import logging
from pyannote.audio import Pipeline
import torch
import whisperx

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

def diarize_audio_hybrid(audio_path: str, device: str = None, model_name: str = "large-v2") -> List[Tuple[str, float, float]]:
    """
    Performs hybrid diarization: initial diarization with WhisperX, then refines with pyannote.audio.

    Args:
        audio_path (str): Path to the preprocessed WAV audio file.
        device (str, optional): Device to use ("cuda" or "cpu"). If None, auto-detect.
        model_name (str, optional): WhisperX model to use.

    Returns:
        List[Tuple[str, float, float]]: List of (speaker_label, start_time, end_time) segments.
    """
    try:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Running WhisperX diarization on {device}...")
        # 1. Transcribe and diarize with WhisperX
        batch_size = 16
        compute_type = "float16" if device == "cuda" else "float32"
        model = whisperx.load_model(model_name, device, compute_type=compute_type)
        result = model.transcribe(audio_path, batch_size=batch_size)
        # Diarization
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=os.getenv('HUGGINGFACE_TOKEN'), device=device)
        diarize_segments = diarize_model(audio_path, result["segments"])
        whisperx_segments = []
        for seg in diarize_segments["segments"]:
            whisperx_segments.append((seg["speaker"], seg["start"], seg["end"]))
        logging.info(f"WhisperX diarization found {len(whisperx_segments)} segments. Refining with pyannote...")
        # 2. Refine with pyannote
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        if not hf_token:
            logging.error("Hugging Face token not found in environment variable 'HUGGINGFACE_TOKEN'.")
            return whisperx_segments
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
        pipeline.to(torch.device(device))
        # Run pyannote on the audio file
        diarization = pipeline(audio_path)
        pyannote_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            pyannote_segments.append((speaker, turn.start, turn.end))
        logging.info(f"pyannote refinement found {len(pyannote_segments)} segments.")
        # Optionally, merge or choose between WhisperX and pyannote segments
        # For now, return pyannote segments as the refined result
        return pyannote_segments
    except Exception as e:
        logging.error(f"Error during hybrid diarization: {e}")
        return [] 