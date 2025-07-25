import streamlit as st
import os
import tempfile
from src import preprocess, diarization, transcribe
from dotenv import load_dotenv
import torch

load_dotenv()

st.title("Multi-Speaker Meeting Transcription & Diarization")
st.write("Upload an audio file (WAV, MP3, etc.) to get a diarized transcript.")

uploaded_file = st.file_uploader("Upload your audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    # Save uploaded file to a temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.audio(tmp_path)
    st.info("Processing audio... This may take a few minutes for long files.")

    # Preprocess
    ok, out = preprocess.preprocess_audio(tmp_path, os.path.dirname(tmp_path))
    if not ok:
        st.error(f"Audio preprocessing failed: {out}")
    else:
        wav_path = out
        # Diarization
        segments = diarization.diarize_audio(wav_path)
        if not segments:
            st.error("Speaker diarization failed or found no speakers.")
        else:
            # Extract required fields from segments for transcription
            try:
                seg_dicts = [
                    {
                        'speaker': seg.get('speaker', 'Unknown'),
                        'start': float(seg.get('start', 0)),
                        'end': float(seg.get('end', 0))
                    }
                    for seg in segments
                ]
            except (ValueError, TypeError) as e:
                st.error(f"Invalid segment data format: {e}")
                seg_dicts = []
            # Transcription
            results = transcribe.transcribe_with_whisperx(
                audio_path=wav_path,
                diarization=seg_dicts,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            if not results:
                st.error("Transcription failed.")
            else:
                # Display transcript
                st.success("Transcription complete!")
                transcript_lines = []
                for r in results:
                    speaker = r.get('speaker', 'Unknown')
                    start = r.get('start', 0)
                    end = r.get('end', 0)
                    text = r.get('text', r.get('error', ''))
                    line = f"[{start:.2f}-{end:.2f}] {speaker}: {text}"
                    transcript_lines.append(line)
                transcript = "\n".join(transcript_lines)
                st.text_area("Diarized Transcript", transcript, height=300)
                st.download_button("Download Transcript", transcript, file_name="transcript.txt")

    # Clean up temp file
    os.remove(tmp_path) 