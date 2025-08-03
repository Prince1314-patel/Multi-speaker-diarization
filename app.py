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

# Add custom CSS for better transcript display
st.markdown("""
<style>
    .transcript-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        line-height: 1.4;
    }
    .speaker-input {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("Multi-Speaker Meeting Transcription & Diarization")
st.write("Upload an audio file (WAV, MP3, etc.) to get a diarized transcript. The processing is done on the server.")

# Cache the transcription results to avoid re-running when speaker names change
@st.cache_data
def process_audio_transcription(audio_file_bytes, file_name):
    """
    Process audio file and return transcription results.
    This function is cached to avoid re-running transcription when speaker names change.
    """
    tmp_path = None
    try:
        # Save uploaded file to a temp location to handle it safely
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as tmp_file:
            tmp_file.write(audio_file_bytes)
            tmp_path = tmp_file.name

        # 1. Preprocess the audio file
        ok, out = preprocess.preprocess_audio(tmp_path, os.path.dirname(tmp_path))
        if not ok:
            return None, f"Audio preprocessing failed: {out}"
        
        wav_path = out
        
        # 2. Perform Speaker Diarization
        diarization_segments, overlap_regions = diarization.diarize_audio(wav_path)
        if not diarization_segments:
            return None, "Speaker diarization failed or found no speakers."

        # 3. Use diarization_segments for transcription
        seg_dicts = diarization_segments
        results = transcribe.transcribe_with_whisperx(
            audio_path=wav_path,
            diarization=seg_dicts,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        if not results:
            return None, "Transcription failed."
        
        return results, None

    except Exception as e:
        return None, f"An unexpected error occurred: {e}"
    finally:
        # Clean up the temporary file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

uploaded_file = st.file_uploader("Upload your audio file", type=["wav", "mp3", "m4a", "ogg", "flac"])

# Add a button to clear cache if needed
col1, col2 = st.columns([3, 1])
with col2:
    if st.button("Clear Cache", help="Clear cached results to re-process the same file"):
        process_audio_transcription.clear()
        st.success("Cache cleared! Upload the file again to re-process.")

if uploaded_file is not None:
    # Display uploaded audio
    st.audio(uploaded_file, format='audio/wav')
    
    # Process audio with caching
    with st.spinner("Processing audio... This may take a few minutes for long files. Please wait."):
        st.info("Step 1/3: Preprocessing audio...")
        st.info("Step 2/3: Performing speaker diarization...")
        st.info("Step 3/3: Transcribing segments...")
        
        results, error = process_audio_transcription(uploaded_file.read(), uploaded_file.name)
        
        if error:
            st.error(error)
            st.stop()
        
        if results:
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
            
            # Create columns for speaker inputs
            cols = st.columns(min(len(unique_speakers), 3))  # Max 3 columns
            
            speaker_names = {}
            for i, speaker in enumerate(unique_speakers):
                col_idx = i % 3
                with cols[col_idx]:
                    speaker_names[speaker] = st.text_input(
                        f"Name for {speaker}:",
                        key=f"speaker_{speaker}",
                        placeholder=f"Enter name for {speaker} (or leave empty for default)"
                    )
            
            # Create speaker mapping
            speaker_mapping = utils.create_speaker_mapping(speaker_names, unique_speakers)
            
            # Post-process transcript
            processed_transcript = utils.post_process_transcript(transcript_segments, speaker_mapping)
            
            # Display processed transcript in a proper text box
            st.markdown("### Processed Transcript")
            
            # Create a container with custom styling for better readability
            transcript_container = st.container()
            with transcript_container:
                # Use text_area for better display with scroll and custom styling
                st.text_area(
                    "Transcript",
                    value=processed_transcript,
                    height=400,
                    disabled=True,
                    help="This is the processed transcript with assigned speaker names",
                    key="transcript_display"
                )
            
            # Download button for processed transcript
            st.download_button(
                "Download Processed Transcript", 
                processed_transcript, 
                file_name="transcript.txt",
                mime="text/plain"
            )
