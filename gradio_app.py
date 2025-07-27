import gradio as gr
import os
import tempfile
import torch
from dotenv import load_dotenv
import shutil

# Load environment variables if you have a .env file
load_dotenv()

# --- User Module Imports ---
try:
    from src import preprocess, diarization, transcribe
except ImportError:
    print("Error: Could not import modules from 'src' directory.")
    print("Please ensure 'preprocess.py', 'diarization.py', and 'transcribe.py' exist in a 'src' folder.")
    exit()

# --- Main Gradio Application Logic ---

# --- MODIFIED FUNCTION SIGNATURE ---
def process_audio_pipeline(audio_path, num_speakers, progress=gr.Progress(track_tqdm=True)):
    """
    This is the core function that processes the uploaded audio file.
    It directly calls the user's modules for each step of the pipeline.
    """
    if not audio_path:
        gr.Warning("Please upload an audio file first!")
        return None, "Error: No audio file uploaded.", "", None

    temp_dir = tempfile.mkdtemp()
    
    try:
        # 1. Preprocess Audio
        progress(0.2, desc="Preprocessing audio...")
        ok, wav_path = preprocess.preprocess_audio(audio_path, temp_dir)
        if not ok:
            raise gr.Error(f"Audio preprocessing failed: {wav_path}")

        # 2. Diarize Audio
        progress(0.4, desc="Diarizing speakers...")
        # --- START OF FIX ---
        # Convert num_speakers from float (Gradio default) to int
        num_speakers_int = int(num_speakers) if num_speakers else 0
        segments = diarization.diarize_audio(wav_path, num_speakers=num_speakers_int)
        # --- END OF FIX ---
        
        if not segments:
            raise gr.Error("Speaker diarization failed or found no speakers.")

        # 3. Extract required fields from segments for transcription
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
        if not results or 'segments' not in results or not results['segments']:
            raise gr.Error("Transcription complete, but no speech was detected in the audio.")

        # 5. Format and Display Transcript
        progress(0.9, desc="Formatting transcript...")
        transcript_lines = []
        for r in results['segments']:
            speaker = r.get('speaker', 'Unknown')
            start = r.get('start', 0)
            end = r.get('end', 0)
            text = r.get('text', '').strip()
            line = f"[{start:.2f}s - {end:.2f}s] {speaker}: {text}"
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
            
            # --- NEW INPUT FIELD ---
            num_speakers_input = gr.Number(
                label="Number of Speakers (Optional)", 
                value=0, 
                info="Leave at 0 for automatic detection.",
                precision=0 # Ensures it's an integer
            )
            
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

    # --- UPDATED .click() FUNCTION ---
    # Connect the button to the processing function
    submit_btn.click(
        fn=process_audio_pipeline,
        inputs=[audio_input, num_speakers_input], # Added num_speakers_input
        outputs=[
            processed_audio_output,
            status_output,
            transcript_output,
            download_output
        ]
    )

if __name__ == "__main__":
    demo.launch(debug=True)
