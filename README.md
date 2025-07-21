# Multi-Speaker Meeting Transcription System

A robust, scalable system for transcribing meetings with multiple speakers, supporting major audio formats (WAV, MP3, etc.), using local OpenAI Whisper for transcription and pyannote.audio for speaker diarization.

## Features
- Accepts WAV, MP3, and other major audio formats
- Speaker diarization and accurate transcription
- Exports results as text, CSV, JSON (optionally SRT/VTT)
- Modular, secure, and future-proof

## Requirements
- Windows 10/11, Python 3.11+
- openai-whisper (local transcription)
- Hugging Face token (for pyannote.audio)

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/multiple-speaker-diarization.git
cd multiple-speaker-diarization
```

### 2. Set Up a Virtual Environment
```bash
python -m venv venv
# On Windows:
./venv/Scripts/activate
# On Unix or MacOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file or set the following environment variables in your shell:

- `HUGGINGFACE_TOKEN` — Your Hugging Face access token (for pyannote.audio)

Example for Windows PowerShell:
```powershell
$env:HUGGINGFACE_TOKEN="your-huggingface-token"
```

Example for .env file:
```
HUGGINGFACE_TOKEN=your-huggingface-token
```

### 5. Project Structure
See [`reference_docs/file_structure.md`](reference_docs/file_structure.md) for the full structure.

### 6. Running the Pipeline
You can run the pipeline by calling the main function in `src/pipeline.py` (to be implemented):

```python
from src.pipeline import run_pipeline
run_pipeline('audio_inputs/your_audio_file.mp3', 'transcripts/')
```

Or create a script to automate the process.

## Transcription Details
This project now uses the local [OpenAI Whisper](https://github.com/openai/whisper) model for all transcription tasks. You can select the model size (e.g., 'base', 'small', 'medium', 'large') in your code. No internet connection or API key is required for transcription, but a compatible GPU is recommended for faster processing.

## Hybrid Diarization Workflow (WhisperX + pyannote)

This project now uses a hybrid diarization approach:

1. **Initial Diarization with WhisperX**: Fast, accurate transcription and speaker segmentation using OpenAI WhisperX.
2. **Refinement with pyannote.audio**: Further refines speaker segments using pyannote's advanced diarization pipeline.

### Requirements
- `whisperx` (latest)
- `pyannote.audio` (latest, installed as a dependency of whisperx)
- Hugging Face account and access token (set as `HUGGINGFACE_TOKEN` environment variable)

### Usage
- The pipeline and app will automatically use the hybrid workflow for diarization.
- No code changes are needed for users; just ensure dependencies are installed and the Hugging Face token is set.

### Notes
- You can adjust the workflow or fallback to either method by editing `src/diarization.py`.
- For best results, use high-quality audio and ensure preprocessing completes successfully.

## Documentation Policy
All documentation, including this README, will be updated automatically with any new feature, change, or modification to the application.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](LICENSE) 