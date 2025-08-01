import streamlit as st
import os
import tempfile
from src import preprocess, diarization, transcribe, utils
from dotenv import load_dotenv
import torch

# It's good practice to handle potential missing modules if this script is shared
try:
    from src import preprocess, diarization, transcribe, utils
except ImportError:
    st.error("Missing source files (preprocess, diarization, transcribe, utils). Make sure they are in a 'src' folder.")
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
            diarization_segments, overlap_regions = diarization.diarize_audio(wav_path)
            if not diarization_segments:
                st.error("Speaker diarization failed or found no speakers.")
                st.stop()

            # 3. Use diarization_segments (already a list of dicts) for transcription
            seg_dicts = diarization_segments
            # No need to convert from list-of-lists; diarization.py returns list of dicts
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
                
                # Extract transcript segments
                transcript_segments = results.get('segments', [])
                if not transcript_segments:
                    st.error("No transcript segments found.")
                    st.stop()
                
                # Extract unique speakers for name input
                unique_speakers = utils.extract_unique_speakers(transcript_segments)
                
                # Speaker name input section
                st.markdown("### Speaker Name Assignment")
                st.write("Assign names to speakers (leave empty to keep default speaker labels):")
                
                speaker_names = {}
                for speaker in unique_speakers:
                    speaker_names[speaker] = st.text_input(
                        f"Name for {speaker}:",
                        key=f"speaker_{speaker}",
                        placeholder=f"Enter name for {speaker} (or leave empty for default)"
                    )
                
                # Create speaker mapping
                speaker_mapping = utils.create_speaker_mapping(speaker_names, unique_speakers)
                
                # Post-process transcript
                processed_transcript = utils.post_process_transcript(transcript_segments, speaker_mapping)
                
                # Display processed transcript
                st.markdown("### Processed Transcript")
                st.markdown(processed_transcript)
                
                # Download button for processed transcript
                st.download_button(
                    "Download Processed Transcript", 
                    processed_transcript, 
                    file_name="transcript.txt",
                    mime="text/plain"
                )

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
    finally:
        # Clean up the temporary file in all cases (success or failure)
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError as e:
                st.warning(f"Could not remove temporary file {tmp_path}: {e}")
