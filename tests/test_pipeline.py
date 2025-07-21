import unittest
import os
from src import pipeline
from dotenv import load_dotenv

load_dotenv()

class TestPipeline(unittest.TestCase):
    """
    Integration test for the full diarization and transcription pipeline.
    This test will invoke all modules and requires valid API keys and internet connection.
    """
    def setUp(self):
        self.sample_wav = 'audio_inputs/harvard-sample-audio.wav'
        self.output_dir = 'transcripts/'

    def test_run_pipeline(self):
        """Test that the pipeline runs end-to-end without error."""
        try:
            pipeline.run_pipeline(self.sample_wav, self.output_dir)
            completed = True
        except Exception as e:
            completed = False
            print(f"Pipeline error: {e}")
        self.assertTrue(completed, "Pipeline did not complete successfully.")

if __name__ == '__main__':
    unittest.main() 