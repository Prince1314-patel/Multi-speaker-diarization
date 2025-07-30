# 🗣️ Multi-Speaker Meeting Transcription System

A robust and scalable system for transcribing meetings involving multiple speakers. Supports major audio formats (WAV, MP3, etc.), using **OpenAI Whisper** for local transcription and **pyannote.audio** for speaker diarization.

---

## 🚀 Features

- ✅ Supports WAV, MP3, and other popular audio formats  
- 🎙️ High-accuracy speaker diarization and transcription  
- 📤 Exports results in `.txt`, `.csv`, `.json` (optionally `.srt` / `.vtt`)  
- 🧩 Modular, secure, and future-ready codebase

---

## 📦 Requirements

- OS: Windows 10/11 or Unix-based systems  
- Python 3.11+  
- Local installation of [OpenAI Whisper](https://github.com/openai/whisper)  
- [Hugging Face Access Token](https://huggingface.co/settings/tokens) (for `pyannote.audio`)  

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/multiple-speaker-diarization.git
cd multiple-speaker-diarization
```

### 2. Create and Activate a Virtual Environment

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

---

## 🔐 Environment Configuration

### Set Your Hugging Face Token

You need a Hugging Face token to access required models from `pyannote.audio`. Create a `.env` file or export it directly in your terminal:

#### Using `.env` file:

```
HUGGINGFACE_TOKEN=your-huggingface-token
```

#### Windows PowerShell:

```powershell
$env:HUGGINGFACE_TOKEN="your-huggingface-token"
```

> ✅ **IMPORTANT:** Visit and authenticate yourself on the following model pages:
>
> * [https://huggingface.co/pyannote/segmentation](https://huggingface.co/pyannote/segmentation)
> * [https://huggingface.co/pyannote/overlapped-speech-detection](https://huggingface.co/pyannote/overlapped-speech-detection)

---

## 📂 Project Structure

Refer to [`reference_docs/file_structure.md`](reference_docs/file_structure.md) for a detailed overview of the directory layout.

---

## ▶️ Running the App (Streamlit Interface)

After setting up everything, you can launch the app using Streamlit:

```bash
cd multiple-speaker-diarization
streamlit run app.py
```

This will start the Streamlit web application in your browser. Use the interface to upload audio files and initiate transcription + speaker diarization.

---

## ▶️ Running the Pipeline Programmatically

You can also use the backend pipeline directly in your scripts:

```python
from src.pipeline import run_pipeline

run_pipeline('audio_inputs/your_audio_file.mp3', 'transcripts/')
```

---

## 🧠 Transcription Engine

This project uses the **local OpenAI Whisper** model for transcription. You can customize the model size by selecting from:

* `base`
* `small`
* `medium`
* `large`

> ⚠️ No internet connection or API key is required for transcription, but a compatible GPU is highly recommended for faster performance.

---

## 🧾 Documentation Policy

Documentation (including this `README.md`) is kept up-to-date with all feature additions or modifications. See the `docs/` directory for extended usage and technical details.

---

## 🧩 Troubleshooting

| Issue                           | Solution                                                               |
| ------------------------------- | ---------------------------------------------------------------------- |
| `pyannote` model download fails | Ensure you’ve authenticated with your Hugging Face token.              |
| CUDA errors or slow performance | Ensure you have a CUDA-compatible GPU and the correct PyTorch install. |
| File format not supported       | Convert your file to `.wav` or `.mp3` before processing.               |

---

## 🤝 Contributing

We welcome contributions from the community!

1. Fork the repository
2. Create a new branch (`git checkout -b feature-xyz`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature-xyz`)
5. Open a Pull Request

> For significant changes, please open an issue to discuss the proposal before PR submission.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
