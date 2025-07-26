import gradio as gr
import os
import tempfile
import torch
from dotenv import load_dotenv

# Load environment variables if you have a .env file
load_dotenv()

# --- User Module Imports ---
# Make sure your 'src' directory with the following modules
# is in the same folder as this Gradio application script.
try:
    from src import preprocess, diarization, transcribe
except ImportError:
    print("Error: Could not import modules from 'src' directory.")
    print("Please ensure 'preprocess.py', 'diarization.py', and 'transcribe.py' exist in a 'src' folder.")
    # Exit if modules are not found, as the app cannot function.
    exit()

# --- Main Gradio Application Logic ---

def process_audio_pipeline(audio_path, progress=gr.Progress(track_tqdm=True)):
    """
    This is the core function that processes the uploaded audio file.
    It directly calls the user's modules for each step of the pipeline.
    """
    if not audio_path:
        gr.Warning("Please upload an audio file first!")
        # Return empty values for all outputs
        return None, "Error: No audio file uploaded.", "", None

    # Use a temporary directory for all processing files, which cleans up automatically
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # 1. Preprocess Audio
            progress(0.2, desc="Preprocessing audio...")
            ok, wav_path = preprocess.preprocess_audio(audio_path, temp_dir)
            if not ok:
                raise gr.Error(f"Audio preprocessing failed: {wav_path}")

            # 2. Diarize Audio
            progress(0.4, desc="Diarizing speakers...")
            segments = diarization.diarize_audio(wav_path)
            if not segments:
                raise gr.Error("Speaker diarization failed or found no speakers.")

            # 3. Extract required fields from segments for transcription
            # This logic is identical to your Streamlit app
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
                raise gr.Error(f"Invalid segment data format from diarization module: {e}")

            # 4. Transcribe Audio
            progress(0.7, desc="Transcribing segments...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            results = transcribe.transcribe_with_whisperx(
                audio_path=wav_path,
                diarization=seg_dicts,
                device=device
            )
            if not results:
                raise gr.Error("Transcription failed.")

            # 5. Format and Display Transcript
            progress(0.9, desc="Formatting transcript...")
            transcript_lines = []
            for r in results:
                speaker = r.get('speaker', 'Unknown')
                start = r.get('start', 0)
                end = r.get('end', 0)
                text = r.get('text', r.get('error', ''))
                line = f"[{start:.2f}-{end:.2f}] {speaker}: {text}"
                transcript_lines.append(line)
            
            final_transcript = "\n".join(transcript_lines)

            # 6. Create a downloadable transcript file
            transcript_filepath = os.path.join(temp_dir, "transcript.txt")
            with open(transcript_filepath, "w", encoding="utf-8") as f:
                f.write(final_transcript)

            # 7. Final update to the UI
            progress(1.0, desc="Complete!")
            return wav_path, "Status: Transcription Complete!", final_transcript, gr.File(value=transcript_filepath, visible=True)

        except Exception as e:
            # Propagate any errors to the Gradio interface
            raise gr.Error(str(e))


# --- Build Gradio Interface ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Multi-Speaker Meeting Transcription & Diarization
        Upload an audio file (WAV, MP3, etc.) to get a diarized transcript.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(type="filepath", label="Upload Your Audio File")
            submit_btn = gr.Button("Start Transcription", variant="primary")
        
        with gr.Column(scale=2):
            status_output = gr.Textbox(label="Current Status", interactive=False)
            processed_audio_output = gr.Audio(label="Processed Audio", interactive=False)

    gr.Markdown("---")
    
    transcript_output = gr.Textbox(
        label="Diarized Transcript", 
        lines=15, 
        interactive=True,
        placeholder="Transcript will appear here..."
    )

    download_output = gr.File(label="Download Transcript", visible=False)

    # Connect the button to the processing function
    submit_btn.click(
        fn=process_audio_pipeline,
        inputs=[audio_input],
        outputs=[
            processed_audio_output,
            status_output,
            transcript_output,
            download_output
        ]
    )

if __name__ == "__main__":
    demo.launch(debug=True, share=True)
