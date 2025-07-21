# Project File Structure

project-root/
│
├── audio_inputs/ # User-supplied meeting audio files (.wav)
├── diarization/ # Intermediate diarization results/segments
├── transcripts/ # Final and intermediate transcription files
│
├── src/ # Source code modules
│ ├── preprocess.py # Audio normalization and conversion scripts
│ ├── diarization.py # Speaker diarization workflow
│ ├── transcribe.py # Groq API transcription logic
│ ├── pipeline.py # Orchestration: combines all steps
│ └── utils.py # Helper functions/utilities
│
├── tests/ # Unit and integration tests, sample data
│
├── requirements.txt # Python dependency list
├── README.md # Project overview and setup guide
├── prd.md # Product Requirements Document
├── implementation_plan.md# Implementation plan/checklist
├── file_structure.md # This structure file