import streamlit as st
import os
import tempfile
from src import preprocess, transcribe
from src.diarization import diarize_audio_nemo
from dotenv import load_dotenv
import json
import csv

load_dotenv()

# Sidebar with project info
st.sidebar.title("About")
st.sidebar.info(
    """
    **Multi-Speaker Meeting Transcription & Diarization**
    
    Powered by NVIDIA NeMo. Upload a meeting audio file (WAV, MP3, etc.) to get a diarized transcript. 
    
    [GitHub](https://github.com/your-username/multiple-speaker-diarization)
    """
)
st.sidebar.markdown("---")
st.sidebar.write("Developed by Your Team")

st.title("🎤 Multi-Speaker Meeting Transcription & Diarization")
st.write("Upload an audio file to get a diarized transcript. All processing is local and private.")

col1, col2 = st.columns([1, 2])

with col1:
    uploaded_file = st.file_uploader("Upload your audio file", type=["wav", "mp3"], help="Supported formats: WAV, MP3. Max 100MB.")
    if uploaded_file is not None:
        st.success(f"File uploaded: {uploaded_file.name}")
        st.write(f"Size: {uploaded_file.size / 1024:.1f} KB")

with col2:
    if uploaded_file is not None:
        # Save uploaded file to a temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        st.audio(tmp_path)

        # Preprocessing
        with st.spinner("Preprocessing audio (format conversion, validation)..."):
            ok, out = preprocess.preprocess_audio(tmp_path, os.path.dirname(tmp_path))
        if not ok:
            st.error(f"Audio preprocessing failed: {out}")
            os.remove(tmp_path)
        else:
            wav_path = out
            # Diarization
            with st.spinner("Running speaker diarization (NVIDIA NeMo)..."):
                segments = diarize_audio_nemo(wav_path)
            if not segments:
                st.error("Speaker diarization failed or found no speakers.")
            else:
                seg_dicts = [
                    {'speaker': s, 'start': float(start), 'end': float(end)}
                    for s, start, end in segments
                ]
                # Transcription
                with st.spinner("Transcribing each speaker segment..."):
                    results = transcribe.transcribe_segments(wav_path, seg_dicts, tempfile.gettempdir())
                if not results:
                    st.error("Transcription failed.")
                else:
                    st.success("Transcription complete!")
                    # Format transcript for display and download
                    transcript_lines = []
                    for r in results:
                        speaker = r.get('speaker', 'Unknown')
                        start = r.get('start', 0)
                        end = r.get('end', 0)
                        text = r.get('text', r.get('error', ''))
                        line = f"[{start:.2f}-{end:.2f}] **{speaker}**: {text}"
                        transcript_lines.append(line)
                    transcript = "\n".join(transcript_lines)

                    with st.expander("Show Diarized Transcript"):
                        st.markdown(transcript, unsafe_allow_html=True)
                        st.text_area("Copy Transcript", transcript, height=200)

                    # Download options
                    st.download_button("Download as TXT", transcript, file_name="transcript.txt")
                    # CSV
                    csv_path = os.path.join(tempfile.gettempdir(), "transcript.csv")
                    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(["speaker", "start", "end", "text"])
                        for r in results:
                            writer.writerow([r.get('speaker', ''), r.get('start', 0), r.get('end', 0), r.get('text', '')])
                    with open(csv_path, "rb") as f:
                        st.download_button("Download as CSV", f, file_name="transcript.csv")
                    # JSON
                    json_path = os.path.join(tempfile.gettempdir(), "transcript.json")
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=2)
                    with open(json_path, "rb") as f:
                        st.download_button("Download as JSON", f, file_name="transcript.json")

        # Clean up temp file
        os.remove(tmp_path) 