import pytest
import os
import json
import tempfile
import numpy as np
import soundfile as sf
import src.pipeline as pipeline

@pytest.fixture
def sample_audio_path():
    """
    Creates a temporary silent WAV audio file (mock data).
    Cleans up only if the file still exists (pipeline may delete it).
    """
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
        sr = 16000
        duration = 5  # seconds
        audio = np.zeros(duration * sr, dtype=np.float32)
        sf.write(temp_audio.name, audio, sr)
        path = temp_audio.name
    yield path
    # Teardown: remove file only if it still exists
    if os.path.exists(path):
        os.unlink(path)

@pytest.fixture
def output_dir():
    """
    Creates a temporary output directory for testing.
    """
    dir_path = tempfile.mkdtemp(prefix="test_output_")
    return dir_path


def test_run_pipeline_with_mocks(sample_audio_path, output_dir, monkeypatch):
    """
    Tests the full pipeline by mocking external dependencies to force output generation.
    """
    # Mock environment variable for Hugging Face token
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "test_token")

    # Mock preprocess_audio to return success and same path
    monkeypatch.setattr(pipeline, 'preprocess_audio', lambda input_path, out_dir: (True, input_path))

    # Mock diarize_audio to return a single speaker segment and no overlaps
    dummy_segments = [{'start': 0.0, 'end': 5.0, 'speaker': 'spk1'}]
    monkeypatch.setattr(pipeline, 'diarize_audio', lambda audio_path, num_speakers: (dummy_segments, []))

    # Mock transcribe_with_whisperx to return a matching transcribed segment
    transcribed_segment = {
        'start': 0.0,
        'end': 5.0,
        'speaker': 'spk1',
        'text': 'hello world',
        'words': []
    }
    monkeypatch.setattr(
        pipeline,
        'transcribe_with_whisperx',
        lambda audio, segments, batch_size: {'segments': [transcribed_segment]}
    )

    # Run the pipeline
    pipeline.run_pipeline(sample_audio_path, output_dir)

    # Derive base name
    base = os.path.splitext(os.path.basename(sample_audio_path))[0]

    # Expected output paths
    json_path = os.path.join(output_dir, f"{base}_transcript.json")
    csv_path = os.path.join(output_dir, f"{base}_transcript.csv")
    txt_path = os.path.join(output_dir, f"{base}_transcript.txt")

    # Assert files exist
    assert os.path.exists(json_path), "Expected JSON transcript not found"
    assert os.path.exists(csv_path), "Expected CSV transcript not found"
    assert os.path.exists(txt_path), "Expected TXT transcript not found"

    # Validate JSON content
    with open(json_path, 'r') as f:
        data = json.load(f)
    assert isinstance(data, list), "JSON transcript should be a list"
    assert data[0]['text'] == 'hello world', "Transcript text mismatch"
