import unittest
from src import diarization
import os
from dotenv import load_dotenv

load_dotenv()

class TestDiarization(unittest.TestCase):
    """
    Unit tests for speaker diarization functions.
    """
    def setUp(self):
        self.sample_wav = 'audio_inputs/harvard-sample-audio.wav'

    def test_diarize_audio(self):
        """Test diarization returns non-empty list of segments with correct structure."""
        segments = diarization.diarize_audio(self.sample_wav)
        self.assertIsInstance(segments, list)
        self.assertTrue(len(segments) > 0, "No segments returned by diarization.")
        for seg in segments:
            self.assertIsInstance(seg, tuple)
            self.assertEqual(len(seg), 3)
            speaker, start, end = seg
            self.assertIsInstance(speaker, str)
            self.assertIsInstance(start, float)
            self.assertIsInstance(end, float)
            self.assertLess(start, end)

if __name__ == '__main__':
    unittest.main() 