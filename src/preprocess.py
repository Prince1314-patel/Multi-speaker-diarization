import os
from typing import Tuple
import soundfile as sf
import librosa
from pydub import AudioSegment
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def validate_audio_file(filepath: str) -> Tuple[bool, str]:
    """
    Validates the input audio file for supported format and reasonable length.

    Args:
        filepath (str): Path to the audio file.

    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    supported_formats = ('.wav', '.mp3')
    if not os.path.isfile(filepath):
        return False, f"File not found: {filepath}"
    if not filepath.lower().endswith(supported_formats):
        return False, f"Unsupported file format: {filepath}"
    try:
        if filepath.lower().endswith('.mp3'):
            audio = AudioSegment.from_mp3(filepath)
            duration_sec = len(audio) / 1000.0
        else:
            info = sf.info(filepath)
            duration_sec = info.duration
        if duration_sec > 3600:
            return False, "Audio file exceeds 60 minutes."
        if os.path.getsize(filepath) > 100 * 1024 * 1024:
            return False, "Audio file exceeds 100MB."
    except Exception as e:
        return False, f"Error reading audio file: {e}"
    return True, ""

def convert_to_wav_mono_16k(input_path: str, output_path: str) -> bool:
    """
    Converts any supported audio file to mono, 16kHz WAV format and saves it.

    Args:
        input_path (str): Path to the input audio file.
        output_path (str): Path to save the converted WAV file.

    Returns:
        bool: True if conversion was successful, False otherwise.
    """
    try:
        logging.info(f"Loading audio file: {input_path}")
        # librosa handles most formats, but fallback to pydub for mp3
        if input_path.lower().endswith('.mp3'):
            audio = AudioSegment.from_mp3(input_path)
            audio = audio.set_channels(1).set_frame_rate(16000)
            audio.export(output_path, format="wav")
            logging.info(f"Converted MP3 to mono 16kHz WAV: {output_path}")
        else:
            y, sr = librosa.load(input_path, sr=16000, mono=True)
            sf.write(output_path, y, 16000)
            logging.info(f"Converted audio to mono 16kHz WAV: {output_path}")
        return True
    except Exception as e:
        logging.error(f"Error converting audio: {e}")
        return False

def preprocess_audio(input_path: str, output_dir: str) -> Tuple[bool, str]:
    """
    Full preprocessing pipeline: validate and convert audio file.

    Args:
        input_path (str): Path to the user-supplied audio file.
        output_dir (str): Directory to save the processed audio.

    Returns:
        Tuple[bool, str]: (success, output_path or error message)
    """
    is_valid, msg = validate_audio_file(input_path)
    if not is_valid:
        logging.error(msg)
        return False, msg
    base = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{base}_mono16k.wav")
    success = convert_to_wav_mono_16k(input_path, output_path)
    if not success:
        return False, f"Failed to convert {input_path} to mono 16kHz WAV."
    return True, output_path 