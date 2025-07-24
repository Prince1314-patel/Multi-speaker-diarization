import pytest
import os
import numpy as np
import soundfile as sf
from src.transcribe import transcribe_with_whisperx

@pytest.fixture(scope="module")
def dummy_audio_file(tmp_path_factory):
    """
    Creates a dummy audio file for testing.
    """
    file_path = tmp_path_factory.mktemp("data") / "dummy_audio.wav"
    samplerate = 16000
    duration = 1  # seconds
    frequency = 440  # Hz
    t = np.linspace(0., duration, int(samplerate * duration), endpoint=False)
    data = 0.5 * np.sin(2. * np.pi * frequency * t)
    sf.write(file_path, data.astype(np.float32), samplerate)
    return str(file_path)

def test_transcribe_with_whisperx_basic(dummy_audio_file):
    """
    Tests the basic functionality of transcribe_with_whisperx.
    """
    # Dummy diarization data (simplified for testing)
    diarization_data = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 0.5},
        {"speaker": "SPEAKER_01", "start": 0.6, "end": 1.0},
    ]

    # Call the transcription function
    # Note: This test will attempt to download WhisperX models if not cached.
    # For actual unit tests, consider mocking whisperx.load_model and its dependencies.
    transcribed_segments = transcribe_with_whisperx(
        audio_path=dummy_audio_file,
        diarization=diarization_data,
        model_name="base",  # Use a slightly larger model
        device="cpu",       # Use CPU for testing to avoid GPU dependency
        batch_size=1,
        compute_type="int8" # Use int8 for faster testing
    )

    # Assertions
    assert isinstance(transcribed_segments, list)
    assert len(transcribed_segments) > 0, "No segments transcribed"

    for segment in transcribed_segments:
        assert "speaker" in segment
        assert "start" in segment
        assert "end" in segment
        assert "text" in segment
        assert "words" in segment
        assert isinstance(segment["words"], list)
        assert len(segment["words"]) > 0, "No words found in segment"

        for word_info in segment["words"]:
            assert "word" in word_info
            assert "start" in word_info
            assert "end" in word_info
