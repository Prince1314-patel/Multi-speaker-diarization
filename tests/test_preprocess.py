import unittest
import os
import tempfile
from src import preprocess

class TestPreprocess(unittest.TestCase):
    """
    Unit tests for audio preprocessing functions.
    """
    def setUp(self):
        self.temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        self.sample_wav = self.temp_audio.name
        self.output_dir = tempfile.mkdtemp()

    def tearDown(self):
        self.temp_audio.close()
        os.unlink(self.temp_audio.name)
        import shutil
        shutil.rmtree(self.output_dir)

    def test_validate_audio_file(self):
        """Test validation of a supported WAV file."""
        is_valid, msg = preprocess.validate_audio_file(self.sample_wav)
        self.assertTrue(is_valid, msg)

    def test_convert_to_wav_mono_16k(self):
        """Test conversion of WAV file to mono 16kHz WAV."""
        output_path = os.path.join(self.output_dir, 'dummy_mono16k.wav')
        success = preprocess.convert_to_wav_mono_16k(self.sample_wav, output_path)
        self.assertTrue(success)
        self.assertTrue(os.path.isfile(output_path))
        if os.path.isfile(output_path):
            os.remove(output_path)

    def test_preprocess_audio(self):
        """Test full preprocessing pipeline."""
        success, result = preprocess.preprocess_audio(self.sample_wav, self.output_dir)
        self.assertTrue(success, result)
        self.assertTrue(os.path.isfile(result))
        if os.path.isfile(result):
            os.remove(result)

if __name__ == '__main__':
    unittest.main() 