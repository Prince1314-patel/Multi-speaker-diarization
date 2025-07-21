from typing import List, Dict
import os
import logging
import requests
import soundfile as sf
import numpy as np
import whisper

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')


def slice_audio_segment(audio_path: str, start: float, end: float, out_path: str) -> bool:
    """
    Slices a segment from the audio file and saves it as a new WAV file.

    Args:
        audio_path (str): Path to the source WAV file.
        start (float): Start time in seconds.
        end (float): End time in seconds.
        out_path (str): Path to save the sliced segment.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        y, sr = sf.read(audio_path)
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segment = y[start_sample:end_sample]
        sf.write(out_path, segment, sr)
        return True
    except Exception as e:
        logging.error(f"Error slicing audio segment: {e}")
        return False


def transcribe_segment(segment_path: str, model_name: str = "base") -> Dict:
    """
    Transcribes an audio segment using the local OpenAI Whisper model.

    Args:
        segment_path (str): Path to the audio segment WAV file.
        model_name (str): Whisper model to use (e.g., 'base', 'small', 'medium', 'large').

    Returns:
        Dict: Transcription result with text and word-level timestamps.
    """
    try:
        model = whisper.load_model(model_name)
        result = model.transcribe(segment_path, word_timestamps=True, verbose=False)
        return result
    except Exception as e:
        logging.error(f"Whisper transcription error: {e}")
        return {"error": str(e)}


def transcribe_segments(audio_path: str, segments: List[Dict], temp_dir: str = "diarization/", model_name: str = "base") -> List[Dict]:
    """
    Transcribes diarized audio segments using the local OpenAI Whisper model.

    Args:
        audio_path (str): Path to the preprocessed WAV audio file.
        segments (List[Dict]): List of diarized segments with 'speaker', 'start', 'end'.
        temp_dir (str): Directory to store temporary segment files.
        model_name (str): Whisper model to use.

    Returns:
        List[Dict]: List of transcribed segments with text and timestamps.
    """
    results = []
    for idx, seg in enumerate(segments):
        speaker = seg['speaker'] if 'speaker' in seg else seg[0]
        start = seg['start'] if 'start' in seg else seg[1]
        end = seg['end'] if 'end' in seg else seg[2]
        seg_path = os.path.join(temp_dir, f"segment_{idx}_{speaker}.wav")
        if not slice_audio_segment(audio_path, start, end, seg_path):
            continue
        logging.info(f"Transcribing segment {idx} ({speaker}): {start}-{end}s")
        result = transcribe_segment(seg_path, model_name)
        result.update({'speaker': speaker, 'start': start, 'end': end})
        results.append(result)
    return results 