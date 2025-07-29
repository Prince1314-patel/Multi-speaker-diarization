 🔊 Multi-Speaker Meeting Transcription System

A robust and scalable transcription system engineered to handle multi-speaker meetings across diverse audio formats. Powered by NVIDIA NeMo for speaker diarization and local transcription, this system offers enterprise-grade performance and modular integration.

---

## 🚀 Key Features

* ✅ **Multi-format Input**: Supports major audio formats including `.wav`, `.mp3`, and more.
* 🧠 **Speaker Diarization**: Utilizes **NVIDIA NeMo**, **WhisperX**, and **pyannote-audio** for accurate speaker segmentation.
* 📝 **High-Precision Transcription**: Combines ASR models with diarization for rich, contextual transcriptions.
* 📤 **Multi-format Export**: Outputs results in `.txt`, `.csv`, `.json`, and optionally `.srt` / `.vtt` for subtitles.
* 🔒 **Secure & Modular**: Built with best practices for maintainability and extensibility.
* ⚙️ **Streamlit Frontend** *(optional)*: Clean UI for quick uploads, visualization, and result export.

---

## 📦 Requirements

> **System**

* OS: Windows 10/11
* Python: v3.11+

> **Python Dependencies**
> Install via `pip install -r requirements.txt`:

```text
nemo_toolkit[asr]
omegaconf
torch
streamlit
numpy
soundfile
librosa
pydub
ffmpeg-python
python-dotenv
requests
rich
pytest
audioread
typing-extensions
jiwer
```

---

## 🌱 Branch Strategy

| Branch            | Description                                                                              |
| ----------------- | ---------------------------------------------------------------------------------------- |
| `dev`             | Implements diarization using **pyannote-audio** only.                                    |
| `hybrid-approach` | Combines **WhisperX** and **pyannote-audio** for enhanced diarization and transcription. |

---
