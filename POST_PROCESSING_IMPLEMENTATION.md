# Post-Processing Implementation for Multi-Speaker Diarization

## Overview

This implementation adds post-processing functionality to the multi-speaker diarization system, allowing users to assign names to detected speakers and download formatted transcripts.

## Features Implemented

### 1. Speaker Name Assignment
- Users can assign names to detected speakers (SPEAKER_00, SPEAKER_01, etc.)
- If no name is provided, the system uses the default speaker label
- Example: SPEAKER_00 → "Prince", SPEAKER_01 → "SPEAKER_01" (default)

### 2. Transcript Formatting
- Timestamps in format: `[start_time - end_time]`
- Speaker names followed by transcribed text
- Clean, readable format suitable for download

### 3. Download Functionality
- Generate downloadable .txt files
- Properly formatted transcripts with user-assigned speaker names

## Implementation Details

### Core Functions (src/utils.py)

#### `post_process_transcript(transcript_segments, speaker_mapping)`
- Takes transcript segments and speaker name mapping
- Replaces speaker labels with user-provided names
- Returns formatted transcript string

#### `extract_unique_speakers(transcript_segments)`
- Extracts unique speaker labels from transcript
- Returns sorted list of speaker identifiers

#### `create_speaker_mapping(speaker_names, unique_speakers)`
- Creates mapping from speaker labels to display names
- Handles empty user inputs by using default speaker labels

### UI Integration

#### Streamlit App (app.py)
- Dynamic speaker name input fields based on detected speakers
- Real-time post-processing after transcription
- Download button for processed transcript

#### Gradio App (gradio_app.py)
- Fixed number of speaker input fields (4 by default)
- Integrated post-processing in pipeline
- Download functionality for processed transcript

## Usage Examples

### Example 1: User Provides Names
```
Input:
- SPEAKER_00 → "Prince"
- SPEAKER_01 → "Alice"

Output:
[0.00s - 3.50s] Prince: Hello, how are you today?
[3.50s - 6.20s] Alice: I'm doing well, thank you.
```

### Example 2: Partial Name Assignment
```
Input:
- SPEAKER_00 → "Prince"
- SPEAKER_01 → (empty - uses default)

Output:
[0.00s - 3.50s] Prince: Hello, how are you today?
[3.50s - 6.20s] SPEAKER_01: I'm doing well, thank you.
```

## File Structure

```
src/
├── utils.py              # Post-processing functions
├── diarization.py        # Speaker detection
├── transcribe.py         # Transcription
└── preprocess.py         # Audio preprocessing

app.py                    # Streamlit interface
gradio_app.py             # Gradio interface
tests/
└── test_post_processing.py  # Unit tests
demo_post_processing.py   # Demo script
```

## Testing

Run the test suite:
```bash
python -m pytest tests/test_post_processing.py -v
```

Run the demo:
```bash
python demo_post_processing.py
```

## Error Handling

- Graceful handling of missing speaker information
- Empty user inputs default to original speaker labels
- Robust processing of various transcript formats
- Comprehensive logging for debugging

## Future Enhancements

1. **Dynamic Speaker Detection**: Automatically detect number of speakers and create input fields
2. **Speaker Name Validation**: Validate user-provided names
3. **Save/Load Mappings**: Persist speaker name mappings between sessions
4. **Advanced Formatting**: Support for different transcript formats (JSON, CSV, etc.)
5. **Batch Processing**: Process multiple audio files with consistent speaker mappings

## API Reference

### `post_process_transcript(transcript_segments, speaker_mapping)`
- **Parameters**:
  - `transcript_segments`: List[Dict] - Transcript segments with speaker, start, end, text
  - `speaker_mapping`: Dict[str, str] - Mapping from speaker labels to display names
- **Returns**: str - Formatted transcript string

### `extract_unique_speakers(transcript_segments)`
- **Parameters**:
  - `transcript_segments`: List[Dict] - Transcript segments
- **Returns**: List[str] - Sorted list of unique speaker labels

### `create_speaker_mapping(speaker_names, unique_speakers)`
- **Parameters**:
  - `speaker_names`: Dict[str, str] - User-provided speaker names
  - `unique_speakers`: List[str] - List of unique speaker labels
- **Returns**: Dict[str, str] - Complete mapping from labels to display names 