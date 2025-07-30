import pytest
import os
import json
import tempfile
from src.pipeline import run_pipeline

@pytest.fixture
def sample_audio_path():
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
        import numpy as np
        import soundfile as sf
        sr = 16000
        duration = 5 # seconds
        audio = np.zeros(duration * sr)
        sf.write(temp_audio.name, audio, sr)
        yield temp_audio.name
        os.unlink(temp_audio.name)

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