import pytest
import os
import json
from src.pipeline import run_pipeline

@pytest.fixture
def sample_audio_path():
    # Create a dummy audio file for testing
    dummy_audio_path = "/tmp/test_audio.wav"
    if not os.path.exists(dummy_audio_path):
        import numpy as np
        import soundfile as sf
        sr = 16000
        duration = 5 # seconds
        audio = np.zeros(duration * sr)
        sf.write(dummy_audio_path, audio, sr)
    return dummy_audio_path

@pytest.fixture
def output_dir():
    # Create a temporary directory for test outputs
    output_dir = "/tmp/test_output"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def test_run_pipeline(sample_audio_path, output_dir):
    """
    Tests the full pipeline to ensure it runs without errors and creates the expected output files.
    """
    run_pipeline(sample_audio_path, output_dir)

    # Check if the output files were created
    base = os.path.splitext(os.path.basename(sample_audio_path))[0]
    json_path = os.path.join(output_dir, f"{base}_diarization.json")
    csv_path = os.path.join(output_dir, f"{base}_diarization.csv")
    txt_path = os.path.join(output_dir, f"{base}_diarization.txt")

    assert os.path.exists(json_path)
    assert os.path.exists(csv_path)
    assert os.path.exists(txt_path)

    # Check if the JSON output is valid
    with open(json_path, 'r') as f:
        data = json.load(f)
    assert isinstance(data, list)