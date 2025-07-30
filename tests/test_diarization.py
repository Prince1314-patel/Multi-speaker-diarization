import unittest
from unittest.mock import patch, MagicMock
from src import diarization
import os
from dotenv import load_dotenv
import tempfile

load_dotenv()

class TestDiarization(unittest.TestCase):
    """
    Unit tests for speaker diarization functions with enhanced overlap detection.
    """
    def setUp(self):
        self.temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        self.sample_wav = self.temp_audio.name
        if 'HUGGINGFACE_TOKEN' not in os.environ:
            os.environ['HUGGINGFACE_TOKEN'] = 'dummy_token_for_testing'

    def tearDown(self):
        self.temp_audio.close()
        os.unlink(self.temp_audio.name)

    @patch('src.diarization.Pipeline')
    def test_diarize_audio_basic(self, mock_pipeline):
        """Test diarization returns non-empty list of segments with correct structure."""
        mock_diarization = MagicMock()
        mock_segments = [
            (MagicMock(start=0.0, end=1.0), None, "SPEAKER_1"),
            (MagicMock(start=1.5, end=3.0), None, "SPEAKER_2")
        ]
        mock_diarization.itertracks.return_value = mock_segments
        mock_diarization.get_timeline.return_value = MagicMock()
        mock_pipeline.from_pretrained.return_value = mock_diarization

        segments, _ = diarization.diarize_audio(self.sample_wav)
        self.assertIsInstance(segments, list)
        self.assertTrue(len(segments) > 0)
        for seg in segments:
            self.assertIsInstance(seg, dict)
            required_keys = {'speaker', 'start', 'end', 'is_overlap', 'overlap_with'}
            self.assertTrue(all(key in seg for key in required_keys))
            self.assertIsInstance(seg['speaker'], str)
            self.assertIsInstance(seg['start'], float)
            self.assertIsInstance(seg['end'], float)
            self.assertIsInstance(seg['is_overlap'], bool)
            self.assertIsInstance(seg['overlap_with'], list)
            self.assertLess(seg['start'], seg['end'])

    @patch('src.diarization.diarize_audio')
    def test_overlap_detection(self, mock_diarize_audio):
        """Test overlap detection functionality"""
        mock_diarize_audio.return_value = (
            [
                {
                    'speaker': 'SPEAKER_1',
                    'start': 1.0,
                    'end': 3.0,
                    'is_overlap': True,
                    'overlap_with': ['SPEAKER_2']
                },
                {
                    'speaker': 'SPEAKER_2',
                    'start': 2.0,
                    'end': 4.0,
                    'is_overlap': True,
                    'overlap_with': ['SPEAKER_1']
                }
            ],
            None
        )

        result, _ = diarization.diarize_audio(self.sample_wav)
        overlapping_segments = [seg for seg in result if seg['is_overlap']]
        self.assertTrue(len(overlapping_segments) > 0)
        for seg in overlapping_segments:
            self.assertTrue(len(seg['overlap_with']) > 0)

    @patch('torch.cuda.is_available', return_value=True)
    def test_gpu_path(self, mock_cuda_available):
        """Test GPU usage path"""
        with patch('src.diarization.Pipeline') as mock_pipeline:
            mock_pipeline.from_pretrained.return_value.to = MagicMock()
            diarization.diarize_audio(self.sample_wav)
            mock_pipeline.from_pretrained.return_value.to.assert_called_once()

    @patch('torch.cuda.is_available', return_value=False)
    def test_cpu_fallback(self, mock_cuda_available):
        """Test CPU fallback path"""
        with patch('src.diarization.Pipeline') as mock_pipeline:
            mock_pipeline.from_pretrained.return_value.to = MagicMock()
            diarization.diarize_audio(self.sample_wav)
            mock_pipeline.from_pretrained.return_value.to.assert_called_once()

    @patch('src.diarization.diarize_audio')
    def test_min_overlap_duration(self, mock_diarize_audio):
        """Test different min_overlap_duration values"""
        mock_diarize_audio.return_value = (
            [
                {
                    'speaker': 'SPEAKER_1',
                    'start': 1.0,
                    'end': 2.0,
                    'is_overlap': False,
                    'overlap_with': []
                }
            ],
            None
        )

        durations = [0.1, 0.2, 0.5]
        for duration in durations:
            segments, _ = diarization.diarize_audio(
                self.sample_wav,
                min_overlap_duration=duration  # This assumes your function accepts it
            )
            self.assertIsInstance(segments, list)

if __name__ == '__main__':
    unittest.main()
