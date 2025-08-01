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
    sf.write(str(file_path), data.astype(np.float32), samplerate)
    return str(file_path)

@pytest.mark.parametrize("model_name,device,compute_type", [
    ("base", "cpu", "int8"),
])
def test_transcribe_with_whisperx_basic(dummy_audio_file, model_name, device, compute_type):
    """
    Tests WhisperX transcription returns expected dict structure and segments.
    """
    # Dummy diarization data
    diar_data = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 0.5},
        {"speaker": "SPEAKER_01", "start": 0.6, "end": 1.0},
    ]

    # Execute transcription
    result = transcribe_with_whisperx(
        audio_path=dummy_audio_file,
        diarization=diar_data,
        model_name=model_name,
        device=device,
        batch_size=1,
        compute_type=compute_type
    )

    # Assert top-level structure
    assert isinstance(result, dict), "Result should be a dict"
    assert "segments" in result, "Result missing 'segments' key"
    segments = result["segments"]
    assert isinstance(segments, list), "'segments' should be a list"
    assert len(segments) > 0, "No segments transcribed"

    # Validate each segment
    for seg in segments:
        assert "speaker" in seg and isinstance(seg["speaker"], str)
        assert "start" in seg and isinstance(seg["start"], (float, int))
        assert "end" in seg and isinstance(seg["end"], (float, int))
        assert "text" in seg and isinstance(seg["text"], str)
        assert "words" in seg and isinstance(seg["words"], list)
        # words list can be empty or contain dicts with 'word','start','end'
        for w in seg["words"]:
            assert "word" in w and "start" in w and "end" in w
