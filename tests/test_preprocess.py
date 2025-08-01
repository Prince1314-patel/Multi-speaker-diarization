import unittest
import os
import tempfile
import numpy as np
import soundfile as sf
from src import preprocess

class TestPreprocess(unittest.TestCase):
    """
    Unit tests for audio preprocessing functions.
    """
    def setUp(self):
        # Create a valid silent WAV file
        self.sample_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        sr = 16000
        duration = 1  # seconds
        audio = np.zeros(sr * duration, dtype=np.float32)
        sf.write(self.sample_wav, audio, sr)

        # Setup output directory
        self.output_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Clean up temp files and directories
        if os.path.exists(self.sample_wav):
            os.remove(self.sample_wav)
        import shutil
        if os.path.isdir(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_validate_audio_file(self):
        """Test validation of a supported WAV file."""
        is_valid, msg = preprocess.validate_audio_file(self.sample_wav)
        self.assertTrue(is_valid, msg)

    def test_convert_to_wav_mono_16k(self):
        """Test conversion of WAV file to mono 16kHz WAV."""
        output_path = os.path.join(self.output_dir, 'dummy_mono16k.wav')
        success = preprocess.convert_to_wav_mono_16k(self.sample_wav, output_path)
        self.assertTrue(success, "Conversion function returned False.")
        self.assertTrue(os.path.isfile(output_path), "Output WAV file not created.")
        # Verify properties
        y, sr = sf.read(output_path)
        self.assertEqual(sr, 16000, "Sample rate is not 16kHz.")
        # Mono means 1D array
        self.assertEqual(len(y.shape), 1, "Output is not mono audio.")

    def test_preprocess_audio(self):
        """Test full preprocessing pipeline."""
        success, result = preprocess.preprocess_audio(self.sample_wav, self.output_dir)
        self.assertTrue(success, result)
        # result should be path to converted file
        self.assertTrue(os.path.isfile(result), "Preprocessed file not found.")
        # Verify output filename suffix
        self.assertTrue(result.endswith('_mono16k.wav'), "Output filename suffix mismatch.")

if __name__ == '__main__':
    unittest.main()
