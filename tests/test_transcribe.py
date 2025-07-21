import unittest
import os
from src import transcribe

class TestTranscribe(unittest.TestCase):
    """
    Unit tests for transcription functions using local OpenAI Whisper model.
    Note: This test requires the openai-whisper package and a compatible local model.
    """
    def setUp(self):
        self.sample_wav = 'audio_inputs/harvard-sample-audio.wav'
        # Use a short segment for testing (first 2 seconds)
        self.segments = [
            {'speaker': 'SPEAKER_0', 'start': 0.0, 'end': 2.0}
        ]
        self.temp_dir = 'diarization/'
        self.model_name = 'base'  # or 'small', 'medium', 'large' as available

    def test_transcribe_segments(self):
        """Test transcription of a short audio segment returns expected structure using local Whisper."""
        results = transcribe.transcribe_segments(self.sample_wav, self.segments, self.temp_dir, self.model_name)
        self.assertIsInstance(results, list)
        self.assertTrue(len(results) > 0, "No transcription results returned.")
        for result in results:
            self.assertIn('speaker', result)
            self.assertIn('start', result)
            self.assertIn('end', result)
            self.assertTrue('text' in result or 'error' in result)

if __name__ == '__main__':
    unittest.main() 