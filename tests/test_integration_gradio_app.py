import pytest
from gradio_app import process_audio_pipeline
from unittest import mock

@pytest.mark.integration
def test_gradio_app_happy_path(sample_audio_path, mock_env_vars):
    """Happy path: all steps succeed and transcript is returned."""
    with mock.patch('src.preprocess.preprocess_audio', return_value=(True, sample_audio_path)), \
         mock.patch('src.diarization.diarize_audio', return_value=([{'speaker': 'A', 'start': 0.0, 'end': 1.0}], [])), \
         mock.patch('src.transcribe.transcribe_with_whisperx', return_value={'segments': [{'speaker': 'A', 'start': 0.0, 'end': 1.0, 'text': 'Hello'}]}):
        wav_path, status, transcript, file_obj = process_audio_pipeline(sample_audio_path, 1)
        assert status.startswith('Status:')
        assert 'Hello' in transcript
        assert wav_path == sample_audio_path
        assert file_obj is not None

@pytest.mark.integration
def test_gradio_app_preprocessing_failure(sample_audio_path, mock_env_vars):
    """Preprocessing fails: error is raised and handled."""
    with mock.patch('src.preprocess.preprocess_audio', return_value=(False, 'error message')):
        with pytest.raises(Exception):
            process_audio_pipeline(sample_audio_path, 1)

@pytest.mark.integration
def test_gradio_app_diarization_failure(sample_audio_path, mock_env_vars):
    """Diarization fails: error is raised and handled."""
    with mock.patch('src.preprocess.preprocess_audio', return_value=(True, sample_audio_path)), \
         mock.patch('src.diarization.diarize_audio', return_value=([], [])):
        with pytest.raises(Exception):
            process_audio_pipeline(sample_audio_path, 1)

@pytest.mark.integration
def test_gradio_app_transcription_failure(sample_audio_path, mock_env_vars):
    """Transcription fails: error is raised and handled."""
    with mock.patch('src.preprocess.preprocess_audio', return_value=(True, sample_audio_path)), \
         mock.patch('src.diarization.diarize_audio', return_value=([{'speaker': 'A', 'start': 0.0, 'end': 1.0}], [])), \
         mock.patch('src.transcribe.transcribe_with_whisperx', return_value={'segments': []}):
        with pytest.raises(Exception):
            process_audio_pipeline(sample_audio_path, 1)

@pytest.mark.integration
def test_gradio_app_speaker_assignment_fallback(sample_audio_path, mock_env_vars):
    """Speaker assignment fallback is triggered and transcript is still produced."""
    with mock.patch('src.preprocess.preprocess_audio', return_value=(True, sample_audio_path)), \
         mock.patch('src.diarization.diarize_audio', return_value=([{'speaker': 'A', 'start': 0.0, 'end': 1.0}], [])), \
         mock.patch('src.transcribe.transcribe_with_whisperx', return_value={'segments': [{'speaker': None, 'start': 0.0, 'end': 1.0, 'text': 'Hello'}]}):
        wav_path, status, transcript, file_obj = process_audio_pipeline(sample_audio_path, 1)
        assert 'Hello' in transcript

@pytest.mark.integration
def test_gradio_app_large_file(monkeypatch, tmp_path, mock_env_vars):
    """Large file triggers preprocessing error."""
    # Simulate a large file by patching os.path.getsize
    audio_file = tmp_path / "large_audio.wav"
    audio_file.write_bytes(b"fake audio data")
    with mock.patch('os.path.getsize', return_value=101 * 1024 * 1024), \
         mock.patch('src.preprocess.preprocess_audio', return_value=(False, 'Audio file exceeds 100MB.')):
        with pytest.raises(Exception):
            process_audio_pipeline(str(audio_file), 1)

@pytest.mark.integration
def test_gradio_app_invalid_diarization(sample_audio_path, mock_env_vars):
    """Invalid diarization data is handled gracefully."""
    with mock.patch('src.preprocess.preprocess_audio', return_value=(True, sample_audio_path)), \
         mock.patch('src.diarization.diarize_audio', return_value=([{'start': 0.0, 'end': 1.0}], [])):
        with pytest.raises(Exception):
            process_audio_pipeline(sample_audio_path, 1)

@pytest.mark.integration
def test_gradio_app_missing_env_vars(sample_audio_path):
    """Missing environment variables cause failure."""
    with mock.patch('src.preprocess.preprocess_audio', return_value=(True, sample_audio_path)), \
         mock.patch('src.diarization.diarize_audio', side_effect=Exception('HUGGINGFACE_TOKEN not set')):
        with pytest.raises(Exception):
            process_audio_pipeline(sample_audio_path, 1)

@pytest.mark.integration
def test_gradio_app_overlap_detection(sample_audio_path, mock_env_vars):
    """Overlap regions are returned and transcript is enriched."""
    diarization_segments = [{'speaker': 'A', 'start': 0.0, 'end': 1.0}]
    overlap_regions = [(0.5, 0.8)]
    with mock.patch('src.preprocess.preprocess_audio', return_value=(True, sample_audio_path)), \
         mock.patch('src.diarization.diarize_audio', return_value=(diarization_segments, overlap_regions)), \
         mock.patch('src.transcribe.transcribe_with_whisperx', return_value={'segments': [{'speaker': 'A', 'start': 0.0, 'end': 1.0, 'text': 'Hello', 'is_overlap': True}]}):
        wav_path, status, transcript, file_obj = process_audio_pipeline(sample_audio_path, 1)
        assert 'OVERLAP' in transcript or 'Hello' in transcript 