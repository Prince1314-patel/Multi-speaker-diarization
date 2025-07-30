import unittest
from unittest.mock import patch, MagicMock
from src import diarization
import os
from dotenv import load_dotenv
import torch

load_dotenv()

class TestDiarization(unittest.TestCase):
    """
    Unit tests for speaker diarization functions with enhanced overlap detection.
    """
    def setUp(self):
        self.sample_wav = 'audio_inputs/harvard-sample-audio.wav'
        # Ensure environment variable exists for testing
        if 'HUGGINGFACE_TOKEN' not in os.environ:
            os.environ['HUGGINGFACE_TOKEN'] = 'dummy_token_for_testing'

    def test_diarize_audio_basic(self):
        """Test diarization returns non-empty list of segments with correct structure."""
        segments, _ = diarization.diarize_audio(self.sample_wav)
        self.assertIsInstance(segments, list)
        self.assertTrue(len(segments) > 0, "No segments returned by diarization.")
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

    def test_overlap_detection(self):
        """Test overlap detection functionality"""
        with patch('src.diarization.Pipeline') as mock_pipeline:
            # Mock pipeline to return overlapping segments
            mock_diarization = MagicMock()
            mock_segments = [
                (MagicMock(start=1.0, end=3.0), None, "SPEAKER_1"),
                (MagicMock(start=2.0, end=4.0), None, "SPEAKER_2")
            ]
            mock_diarization.itertracks.return_value = mock_segments
            mock_diarization.get_timeline.return_value = MagicMock()
            mock_pipeline.from_pretrained.return_value = mock_diarization
            
            result = diarization.diarize_audio(self.sample_wav)
            
            # Verify overlap detection
            overlapping_segments = [seg for seg in result if seg['is_overlap']]
            self.assertTrue(len(overlapping_segments) > 0)
            
            # Verify overlap information
            for seg in overlapping_segments:
                self.assertTrue(len(seg['overlap_with']) > 0)

    @patch('torch.cuda.is_available')
    def test_gpu_usage(self, mock_cuda_available):
        """Test GPU/CPU handling"""
        # Test GPU path
        mock_cuda_available.return_value = True
        with patch('src.diarization.Pipeline') as mock_pipeline:
            diarization.diarize_audio(self.sample_wav)
            mock_pipeline.from_pretrained.return_value.to.assert_called_once()
        
        # Test CPU fallback
        mock_cuda_available.return_value = False
        with patch('src.diarization.Pipeline') as mock_pipeline:
            diarization.diarize_audio(self.sample_wav)
            mock_pipeline.from_pretrained.return_value.to.assert_called_once()

    def test_min_overlap_duration(self):
        """Test minimum overlap duration parameter"""
        # Test with different min_overlap_duration values
        durations = [0.1, 0.2, 0.5]
        for duration in durations:
            segments = diarization.diarize_audio(
                self.sample_wav, 
                min_overlap_duration=duration
            )
            self.assertIsInstance(segments, list)

if __name__ == '__main__':
    unittest.main() 