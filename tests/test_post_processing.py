import pytest
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import post_process_transcript, extract_unique_speakers, create_speaker_mapping

class TestPostProcessing:
    """Test cases for post-processing functionality."""
    
    def test_extract_unique_speakers(self):
        """Test extracting unique speakers from transcript segments."""
        # Mock transcript segments
        segments = [
            {'speaker': 'SPEAKER_00', 'start': 0, 'end': 5, 'text': 'Hello'},
            {'speaker': 'SPEAKER_01', 'start': 5, 'end': 10, 'text': 'Hi there'},
            {'speaker': 'SPEAKER_00', 'start': 10, 'end': 15, 'text': 'How are you?'},
            {'speaker': 'SPEAKER_02', 'start': 15, 'end': 20, 'text': 'Good, thanks'}
        ]
        
        unique_speakers = extract_unique_speakers(segments)
        expected = ['SPEAKER_00', 'SPEAKER_01', 'SPEAKER_02']
        
        assert unique_speakers == expected
    
    def test_create_speaker_mapping(self):
        """Test creating speaker mapping with user names."""
        unique_speakers = ['SPEAKER_00', 'SPEAKER_01', 'SPEAKER_02']
        speaker_names = {
            'SPEAKER_00': 'Prince',
            'SPEAKER_01': '',  # Empty name should use default
            'SPEAKER_02': 'Alice'
        }
        
        mapping = create_speaker_mapping(speaker_names, unique_speakers)
        expected = {
            'SPEAKER_00': 'Prince',
            'SPEAKER_01': 'SPEAKER_01',  # Default when no name provided
            'SPEAKER_02': 'Alice'
        }
        
        assert mapping == expected
    
    def test_post_process_transcript(self):
        """Test post-processing transcript with speaker names."""
        segments = [
            {'speaker': 'SPEAKER_00', 'start': 0.0, 'end': 5.0, 'text': 'Hello'},
            {'speaker': 'SPEAKER_01', 'start': 5.0, 'end': 10.0, 'text': 'Hi there'},
            {'speaker': 'SPEAKER_00', 'start': 10.0, 'end': 15.0, 'text': 'How are you?'}
        ]
        
        speaker_mapping = {
            'SPEAKER_00': 'Prince',
            'SPEAKER_01': 'SPEAKER_01'  # Default name
        }
        
        result = post_process_transcript(segments, speaker_mapping)
        
        expected_lines = [
            '[0.00s - 5.00s] Prince: Hello',
            '[5.00s - 10.00s] SPEAKER_01: Hi there',
            '[10.00s - 15.00s] Prince: How are you?'
        ]
        expected = '\n'.join(expected_lines)
        
        assert result == expected
    
    def test_post_process_empty_segments(self):
        """Test post-processing with empty segments."""
        segments = []
        speaker_mapping = {}
        
        result = post_process_transcript(segments, speaker_mapping)
        assert result == ""
    
    def test_post_process_missing_speaker(self):
        """Test post-processing with segments missing speaker information."""
        segments = [
            {'speaker': None, 'start': 0.0, 'end': 5.0, 'text': 'Hello'},
            {'speaker': 'SPEAKER_00', 'start': 5.0, 'end': 10.0, 'text': 'Hi there'}
        ]
        
        speaker_mapping = {'SPEAKER_00': 'Prince'}
        
        result = post_process_transcript(segments, speaker_mapping)
        
        # Should handle None speaker gracefully
        assert '[0.00s - 5.00s] Unknown: Hello' in result
        assert '[5.00s - 10.00s] Prince: Hi there' in result

if __name__ == "__main__":
    pytest.main([__file__]) 