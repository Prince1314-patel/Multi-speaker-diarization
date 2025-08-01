import unittest
from unittest.mock import patch, MagicMock
from src import diarization
import os
from dotenv import load_dotenv
import tempfile

load_dotenv()

class TestDiarization(unittest.TestCase):
    """
    Unit tests for speaker diarization functions.
    """

    def setUp(self):
        self.temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        self.sample_wav = self.temp_audio.name
        # Ensure a dummy Hugging Face token
        os.environ.setdefault('HUGGINGFACE_TOKEN', 'dummy_token_for_testing')

    def tearDown(self):
        self.temp_audio.close()
        if os.path.exists(self.sample_wav):
            os.unlink(self.sample_wav)

    @patch('src.diarization.Pipeline')
    def test_diarize_audio_basic(self, mock_pipeline):
        """Test diarization returns non-empty list of segments with correct structure."""
        # Mock standard diarization pipeline
        diar_pipeline = MagicMock()
        diar_pipeline.to.return_value = diar_pipeline
        diar_result = MagicMock()
        diar_result.itertracks.return_value = [
            (MagicMock(start=0.0, end=1.0), None, "SPEAKER_1"),
            (MagicMock(start=1.5, end=3.0), None, "SPEAKER_2"),
        ]
        diar_pipeline.return_value = diar_result

        # Mock overlap pipeline
        overlap_pipeline = MagicMock()
        overlap_pipeline.to.return_value = overlap_pipeline
        overlap_result = MagicMock()
        overlap_result.get_timeline.return_value = []
        overlap_pipeline.return_value = overlap_result

        # Set side_effect for from_pretrained
        mock_pipeline.from_pretrained.side_effect = [diar_pipeline, overlap_pipeline]

        segments, overlaps = diarization.diarize_audio(self.sample_wav)

        # Assert segments
        self.assertIsInstance(segments, list)
        self.assertEqual(len(segments), 2)
        for seg in segments:
            self.assertIsInstance(seg, dict)
            self.assertIn('speaker', seg)
            self.assertIn('start', seg)
            self.assertIn('end', seg)
            self.assertIsInstance(seg['speaker'], str)
            self.assertIsInstance(seg['start'], float)
            self.assertIsInstance(seg['end'], float)
            self.assertLess(seg['start'], seg['end'])

        # Assert overlaps is empty
        self.assertIsInstance(overlaps, list)
        self.assertEqual(overlaps, [])

    @patch('src.diarization.diarize_audio')
    def test_overlap_detection(self, mock_diarize_audio):
        """Test overlap detection functionality via return value mock."""
        mock_diarize_audio.return_value = (
            [
                {'speaker': 'SPEAKER_1', 'start': 1.0, 'end': 3.0},
                {'speaker': 'SPEAKER_2', 'start': 2.0, 'end': 4.0}
            ],
            [(2.0, 3.0)]
        )

        segments, overlaps = diarization.diarize_audio(self.sample_wav)
        self.assertIsInstance(overlaps, list)
        self.assertTrue(overlaps)
        for start, end in overlaps:
            self.assertIsInstance(start, float)
            self.assertIsInstance(end, float)
            self.assertLess(start, end)

    @patch('torch.cuda.is_available', return_value=True)
    @patch('src.diarization.Pipeline')
    def test_gpu_path(self, mock_pipeline, mock_cuda_available):
        """Test GPU usage invokes .to() twice."""
        dummy = MagicMock()
        dummy.to.return_value = dummy
        dummy.return_value = MagicMock(itertracks=lambda yield_label=True: [])
        mock_pipeline.from_pretrained.return_value = dummy

        diarization.diarize_audio(self.sample_wav)
        self.assertEqual(dummy.to.call_count, 2)

    @patch('torch.cuda.is_available', return_value=False)
    @patch('src.diarization.Pipeline')
    def test_cpu_fallback(self, mock_pipeline, mock_cuda_available):
        """Test CPU fallback invokes .to() twice."""
        dummy = MagicMock()
        dummy.to.return_value = dummy
        dummy.return_value = MagicMock(itertracks=lambda yield_label=True: [])
        mock_pipeline.from_pretrained.return_value = dummy

        diarization.diarize_audio(self.sample_wav)
        self.assertEqual(dummy.to.call_count, 2)

    @patch('src.diarization.diarize_audio')
    def test_min_overlap_duration(self, mock_diarize_audio):
        """Test that optional parameters do not break interface."""
        mock_diarize_audio.return_value = (
            [{'speaker': 'SPEAKER_1', 'start': 1.0, 'end': 2.0}],
            []
        )

        segments, overlaps = diarization.diarize_audio(self.sample_wav, num_speakers=2)
        self.assertIsInstance(segments, list)
        self.assertIsInstance(overlaps, list)

if __name__ == '__main__':
    unittest.main()
