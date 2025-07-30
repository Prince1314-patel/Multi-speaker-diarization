import pytest
from unittest import mock
import app

@pytest.mark.integration
def test_streamlit_app_happy_path(sample_audio_path, mock_env_vars):
    """Happy path: all steps succeed and transcript is returned."""
    with mock.patch('app.preprocess.preprocess_audio', return_value=(True, sample_audio_path)), \
         mock.patch('app.diarization.diarize_audio', return_value=([{'speaker': 'A', 'start': 0.0, 'end': 1.0}], [])), \
         mock.patch('app.transcribe.transcribe_with_whisperx', return_value={'segments': [{'speaker': 'A', 'start': 0.0, 'end': 1.0, 'text': 'Hello'}]}):
        ok, out = app.preprocess.preprocess_audio(sample_audio_path, '/tmp')
        assert ok
        diarization_segments, overlap_regions = app.diarization.diarize_audio(out)
        assert diarization_segments
        seg_dicts = diarization_segments
        results = app.transcribe.transcribe_with_whisperx(
            audio_path=out,
            diarization=seg_dicts,
            device='cpu'
        )
        assert 'segments' in results
        assert results['segments'][0]['text'] == 'Hello'

@pytest.mark.integration
def test_streamlit_app_preprocessing_failure(sample_audio_path, mock_env_vars):
    """Preprocessing fails: error is raised and handled."""
    with mock.patch('app.preprocess.preprocess_audio', return_value=(False, 'error message')):
        ok, out = app.preprocess.preprocess_audio(sample_audio_path, '/tmp')
        assert not ok
        assert out == 'error message'

@pytest.mark.integration
def test_streamlit_app_diarization_failure(sample_audio_path, mock_env_vars):
    """Diarization fails: error is raised and handled."""
    with mock.patch('app.preprocess.preprocess_audio', return_value=(True, sample_audio_path)), \
         mock.patch('app.diarization.diarize_audio', return_value=([], [])):
        diarization_segments, overlap_regions = app.diarization.diarize_audio(sample_audio_path)
        assert not diarization_segments

@pytest.mark.integration
def test_streamlit_app_transcription_failure(sample_audio_path, mock_env_vars):
    """Transcription fails: error is raised and handled."""
    with mock.patch('app.preprocess.preprocess_audio', return_value=(True, sample_audio_path)), \
         mock.patch('app.diarization.diarize_audio', return_value=([{'speaker': 'A', 'start': 0.0, 'end': 1.0}], [])), \
         mock.patch('app.transcribe.transcribe_with_whisperx', return_value={'segments': []}):
        results = app.transcribe.transcribe_with_whisperx(
            audio_path=sample_audio_path,
            diarization=[{'speaker': 'A', 'start': 0.0, 'end': 1.0}],
            device='cpu'
        )
        assert results['segments'] == []

@pytest.mark.integration
def test_streamlit_app_speaker_assignment_fallback(sample_audio_path, mock_env_vars):
    """Speaker assignment fallback is triggered and transcript is still produced."""
    with mock.patch('app.preprocess.preprocess_audio', return_value=(True, sample_audio_path)), \
         mock.patch('app.diarization.diarize_audio', return_value=([{'speaker': 'A', 'start': 0.0, 'end': 1.0}], [])), \
         mock.patch('app.transcribe.transcribe_with_whisperx', return_value={'segments': [{'speaker': None, 'start': 0.0, 'end': 1.0, 'text': 'Hello'}]}):
        results = app.transcribe.transcribe_with_whisperx(
            audio_path=sample_audio_path,
            diarization=[{'speaker': 'A', 'start': 0.0, 'end': 1.0}],
            device='cpu'
        )
        assert results['segments'][0]['text'] == 'Hello'

@pytest.mark.integration
def test_streamlit_app_large_file(monkeypatch, tmp_path, mock_env_vars):
    """Large file triggers preprocessing error."""
    audio_file = tmp_path / "large_audio.wav"
    audio_file.write_bytes(b"fake audio data")
    with mock.patch('os.path.getsize', return_value=101 * 1024 * 1024), \
         mock.patch('app.preprocess.preprocess_audio', return_value=(False, 'Audio file exceeds 100MB.')):
        ok, out = app.preprocess.preprocess_audio(str(audio_file), '/tmp')
        assert not ok
        assert out == 'Audio file exceeds 100MB.'

@pytest.mark.integration
def test_streamlit_app_invalid_diarization(sample_audio_path, mock_env_vars):
    """Invalid diarization data is handled gracefully."""
    with mock.patch('app.preprocess.preprocess_audio', return_value=(True, sample_audio_path)), \
         mock.patch('app.diarization.diarize_audio', return_value=([{'start': 0.0, 'end': 1.0}], [])):
        diarization_segments, overlap_regions = app.diarization.diarize_audio(sample_audio_path)
        assert 'speaker' not in diarization_segments[0]

@pytest.mark.integration
def test_streamlit_app_missing_env_vars(sample_audio_path):
    """Missing environment variables cause failure."""
    with mock.patch('app.preprocess.preprocess_audio', return_value=(True, sample_audio_path)), \
         mock.patch('app.diarization.diarize_audio', side_effect=Exception('HUGGINGFACE_TOKEN not set')):
        with pytest.raises(Exception):
            app.diarization.diarize_audio(sample_audio_path)

@pytest.mark.integration
def test_streamlit_app_overlap_detection(sample_audio_path, mock_env_vars):
    """Overlap regions are returned and transcript is enriched."""
    diarization_segments = [{'speaker': 'A', 'start': 0.0, 'end': 1.0}]
    overlap_regions = [(0.5, 0.8)]
    with mock.patch('app.preprocess.preprocess_audio', return_value=(True, sample_audio_path)), \
         mock.patch('app.diarization.diarize_audio', return_value=(diarization_segments, overlap_regions)), \
         mock.patch('app.transcribe.transcribe_with_whisperx', return_value={'segments': [{'speaker': 'A', 'start': 0.0, 'end': 1.0, 'text': 'Hello', 'is_overlap': True}]}):
        results = app.transcribe.transcribe_with_whisperx(
            audio_path=sample_audio_path,
            diarization=diarization_segments,
            device='cpu'
        )
        assert results['segments'][0]['is_overlap'] 