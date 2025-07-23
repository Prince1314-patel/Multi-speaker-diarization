# Multi-Speaker Meeting Transcription System

A robust, scalable system for transcribing meetings with multiple speakers, supporting major audio formats (WAV, MP3, etc.), using NVIDIA NeMo for speaker diarization and local transcription.

## Features
- Accepts WAV, MP3, and other major audio formats
- Speaker diarization (NVIDIA NeMo) and accurate transcription
- Exports results as text, CSV, JSON (optionally SRT/VTT)
- Modular, secure, and future-proof

## Requirements
- Windows 10/11, Python 3.11+
- NVIDIA NeMo (nemo_toolkit[asr])
- omegaconf
- torch
- streamlit
- numpy, soundfile, librosa, pydub, ffmpeg-python, python-dotenv, requests, rich, pytest, audioread, typing-extensions, jiwer

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

### 4. Project Structure
See [`reference_docs/file_structure.md`](reference_docs/file_structure.md) for the full structure.

### 5. Running the Pipeline
You can run the pipeline by calling the main function in `src/pipeline.py`:

```python
from src.pipeline import run_pipeline
run_pipeline('audio_inputs/your_audio_file.mp3', 'transcripts/')
```

Or create a script to automate the process.

## Diarization and Transcription Details
This project uses [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) for all speaker diarization tasks. The diarization config YAML file should be placed in the `src/` directory (e.g., `diar_infer_general.yaml`).

Transcription is performed locally on each diarized segment. No cloud API or external service is required for diarization or transcription.

## Evaluation Scripts

To quantitatively assess diarization and transcription quality, use the evaluation scripts in the `evaluation/` directory:

### Diarization Error Rate (DER)
- **Script:** `evaluation/evaluate_der.py`
- **Usage:**
  ```bash
  python evaluation/evaluate_der.py --reference reference.rttm --hypothesis hypothesis.rttm
  ```
- **Description:** Compares your diarization output (RTTM format) to a ground-truth RTTM and prints DER and error breakdown.

### Word Error Rate (WER)
- **Script:** `evaluation/evaluate_wer.py`
- **Usage:**
  ```bash
  python evaluation/evaluate_wer.py --reference reference.txt --hypothesis hypothesis.txt
  ```
- **Description:** Compares your transcript to a ground-truth transcript and prints WER and error details.

## Documentation Policy
All documentation, including this README, will be updated automatically with any new feature, change, or modification to the application.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](LICENSE) 