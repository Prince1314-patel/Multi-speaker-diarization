import streamlit as st
import os
import tempfile
from src import preprocess, diarization, transcribe
from dotenv import load_dotenv
import torch

# It's good practice to handle potential missing modules if this script is shared
try:
    from src import preprocess, diarization, transcribe
except ImportError:
    st.error("Missing source files (preprocess, diarization, transcribe). Make sure they are in a 'src' folder.")
    st.stop()


load_dotenv()

st.set_page_config(layout="wide")
st.title("Multi-Speaker Meeting Transcription & Diarization")
st.write("Upload an audio file (WAV, MP3, etc.) to get a diarized transcript. The processing is done on the server.")

uploaded_file = st.file_uploader("Upload your audio file", type=["wav", "mp3", "m4a", "ogg", "flac"])

if uploaded_file is not None:
    tmp_path = None
    try:
        # Save uploaded file to a temp location to handle it safely
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        st.audio(tmp_path, format='audio/wav')
        
        with st.spinner("Processing audio... This may take a few minutes for long files. Please wait."):
            # 1. Preprocess the audio file
            st.info("Step 1/3: Preprocessing audio...")
            ok, out = preprocess.preprocess_audio(tmp_path, os.path.dirname(tmp_path))
            if not ok:
                st.error(f"Audio preprocessing failed: {out}")
                st.stop()
            
            wav_path = out
            
            # 2. Perform Speaker Diarization
            st.info("Step 2/3: Performing speaker diarization...")
            segments = diarization.diarize_audio(wav_path)
            if not segments:
                st.error("Speaker diarization failed or found no speakers.")
                st.stop()

            # 3. Process segments and transcribe
            st.info("Step 3/3: Transcribing audio segments...")
            seg_dicts = []
            try:
                # FIX: The traceback shows an "AttributeError: 'list' object has no attribute 'get'".
                # This happens because the 'segments' variable is a list of lists (e.g., [['SPEAKER_00', 1.23, 4.56], ...])
                # instead of a list of dictionaries, which the original code expected.
                # This change in output format is likely due to the pyannote.audio version mismatch mentioned in the warnings.
                # We now access the data by index (seg[0], seg[1], seg[2]) instead of by key (seg.get('key')).
                seg_dicts = [
                    {
                        'speaker': seg[0],
                        'start': float(seg[1]),
                        'end': float(seg[2])
                    }
                    for seg in segments
                ]
            except (ValueError, TypeError, IndexError) as e:
                st.error(f"Error parsing segment data: {e}")
                st.warning("The format of the diarization output may have changed. Displaying the first few segments for debugging:")
                st.code(str(segments[:5]))
                st.stop()
            
            if seg_dicts:
                # Transcription using whisperX
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
                        # Use .get() with defaults for robust dictionary access from transcription results
                        speaker = r.get('speaker', 'Unknown')
                        start = r.get('start', 0)
                        end = r.get('end', 0)
                        text = r.get('text', r.get('error', ''))
                        line = f"[{float(start):.2f}-{float(end):.2f}] **{speaker}**: {text}"
                        transcript_lines.append(line)
                    
                    transcript = "\n\n".join(transcript_lines)
                    st.markdown("### Diarized Transcript")
                    st.markdown(transcript)
                    st.download_button("Download Transcript", transcript, file_name="transcript.txt")

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
    finally:
        # Clean up the temporary file in all cases (success or failure)
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError as e:
                st.warning(f"Could not remove temporary file {tmp_path}: {e}")
