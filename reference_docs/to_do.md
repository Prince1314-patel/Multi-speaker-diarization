# Project To-Do List

## 1. Environment Setup
- [x] Install Python 3.11 (Windows 10/11, 64-bit)
- [x] Create and activate a virtual environment
- [x] Install dependencies: `groq`, `pyannote.audio`, `soundfile`, `librosa`, etc.
- [x] Set up Groq API key and Hugging Face token securely

## 2. Audio Preprocessing
- [x] Normalize audio files to mono, 16kHz
- [x] Validate input file format and length

## 3. Speaker Diarization
- [x] Integrate `pyannote.audio` for speaker segmentation
- [x] Generate (speaker label, start time, end time) segments

## 4. Speech Transcription
- [x] Slice audio by diarized segments
- [x] Transcribe each segment using Groq Whisper API
- [x] Retrieve text and word-level timestamps

## 5. Result Aggregation
- [x] Combine transcripts by time and speaker
- [x] Build structured data: speaker label, start, end, transcript

## 6. Output and Export
- [x] Format results as CSV, JSON, and readable text transcript
- [ ] (Optional) Export SRT/VTT subtitle tracks

## 7. Error Handling & Logging
- [x] Implement status/error logs for API calls, audio slicing, and diarization

## 8. Security
- [x] Ensure API keys/tokens are never stored in code or public repos
- [x] Document secure token setup for users

## 9. Testing and Validation
- [ ] Use test audio files to validate system accuracy
- [ ] Measure word error rate (WER) and diarization accuracy
- [ ] Adjust pipeline for improved accuracy as needed

## 10. Future Enhancements
- [ ] Add real-time audio streaming support
- [ ] Implement known-speaker enrollment and identification
- [ ] Develop UI/dashboard for uploads and result browsing

---

### Pipeline Orchestration
- [x] Implemented main pipeline orchestration in `src/pipeline.py` to coordinate preprocessing, diarization, transcription, aggregation, and export with robust logging and error handling.

### Reference: File Structure
- Organize code and data as described in `file_structure.md` (audio_inputs/, diarization/, transcripts/, src/, tests/, etc.)
- Ensure modularity for future enhancements

### Reference: PRD
- Meet accuracy, latency, and export requirements
- Support multiple languages (primary: English)
- Ensure system runs on Windows 10/11, Python 3.11+
- Handle files up to 100MB or 60 minutes 