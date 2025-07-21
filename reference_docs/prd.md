# Product Requirements Document (PRD)

## Project Title:
Multi-Speaker Meeting Transcription System (Windows, Groq Whisper + Pyannote)

## Purpose
Build a robust, scalable system to transcribe meetings recorded as audio files on Windows. The system must:
- Accurately transcribe speech for multiple speakers.
- Attribute each speaker’s contributions.
- Operate efficiently using Groq Whisper (API) and pyannote.audio (local, diarization).

## Target Users
- Enterprise professionals and businesses
- Remote teams needing searchable, attributed meeting transcriptions

## Key Features
- Accepts meeting audio files (WAV, 16kHz mono recommended)
- Uses Groq API (Whisper Large-v3) for state-of-the-art speech recognition
- Utilizes pyannote.audio for automatic speaker diarization
- Outputs transcriptions indicating speaker names (labels), timestamps, and text
- Supports multiple languages (primary: English; fallback: auto-detect)
- Provides organized, exportable results (text/CSV/JSON)

## Non-Functional Requirements
- Runs on Windows 10/11 with Python 3.11+
- No GPU required (for Groq API); pyannote runs on CPU but supports GPU
- Secure API and token management (Groq, Hugging Face)
- Modular architecture for future real-time (streaming) support
- Handles audio files up to 100MB or 60 minutes

## Constraints
- Requires internet for GroqCloud transcription
- Diarization accuracy depends on audio quality
- Speaker identification labels are generic unless enriched with enrollment

## Success Metrics
- ≥90% word-level accuracy on clear audio
- ≤10% diarization error on clean multi-speaker files
- End-to-end processing latency ≤10 minutes for 1-hour meeting (offline + API time)
