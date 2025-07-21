from typing import List, Dict
import os
import logging
import requests
import soundfile as sf
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

GROQ_API_URL = "https://api.groq.com/openai/v1/audio/transcriptions"


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


def transcribe_segment(segment_path: str, api_key: str) -> Dict:
    """
    Sends an audio segment to the Groq Whisper API for transcription.

    Args:
        segment_path (str): Path to the audio segment WAV file.
        api_key (str): Groq API key.

    Returns:
        Dict: Transcription result with text and word-level timestamps.
    """
    try:
        with open(segment_path, 'rb') as f:
            files = {'file': (os.path.basename(segment_path), f, 'audio/wav')}
            data = {
                'model': 'whisper-large-v3',
                'response_format': 'verbose_json',
                'timestamp_granularities[]': ['word']
            }
            headers = {'Authorization': f'Bearer {api_key}'}
            response = requests.post(GROQ_API_URL, files=files, data=data, headers=headers)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logging.error(f"Groq API transcription error: {e}")
        return {"error": str(e)}


def transcribe_segments(audio_path: str, segments: List[Dict], temp_dir: str = "diarization/") -> List[Dict]:
    """
    Transcribes diarized audio segments using Groq Whisper API.

    Args:
        audio_path (str): Path to the preprocessed WAV audio file.
        segments (List[Dict]): List of diarized segments with 'speaker', 'start', 'end'.
        temp_dir (str): Directory to store temporary segment files.

    Returns:
        List[Dict]: List of transcribed segments with text and timestamps.
    """
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        logging.error("Groq API key not found in environment variable 'GROQ_API_KEY'.")
        return []
    results = []
    for idx, seg in enumerate(segments):
        speaker = seg['speaker'] if 'speaker' in seg else seg[0]
        start = seg['start'] if 'start' in seg else seg[1]
        end = seg['end'] if 'end' in seg else seg[2]
        seg_path = os.path.join(temp_dir, f"segment_{idx}_{speaker}.wav")
        if not slice_audio_segment(audio_path, start, end, seg_path):
            continue
        logging.info(f"Transcribing segment {idx} ({speaker}): {start}-{end}s")
        result = transcribe_segment(seg_path, api_key)
        result.update({'speaker': speaker, 'start': start, 'end': end})
        results.append(result)
    return results 