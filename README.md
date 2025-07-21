# Multi-Speaker Meeting Transcription System

A robust, scalable system for transcribing meetings with multiple speakers, supporting major audio formats (WAV, MP3, etc.), using Groq Whisper API for transcription and pyannote.audio for speaker diarization.

## Features
- Accepts WAV, MP3, and other major audio formats
- Speaker diarization and accurate transcription
- Exports results as text, CSV, JSON (optionally SRT/VTT)
- Modular, secure, and future-proof

## Requirements
- Windows 10/11, Python 3.11+
- Internet connection for GroqCloud transcription
- Groq API key and Hugging Face token

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

- `GROQ_API_KEY` — Your Groq Whisper API key
- `HUGGINGFACE_TOKEN` — Your Hugging Face access token (for pyannote.audio)

Example for Windows PowerShell:
```powershell
$env:GROQ_API_KEY="your-groq-api-key"
$env:HUGGINGFACE_TOKEN="your-huggingface-token"
```

Example for .env file:
```
GROQ_API_KEY=your-groq-api-key
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

## Documentation Policy
All documentation, including this README, will be updated automatically with any new feature, change, or modification to the application.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](LICENSE) 