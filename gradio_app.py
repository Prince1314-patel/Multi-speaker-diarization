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
    from src import preprocess, diarization, transcribe, utils
except ImportError:
    print("Error: Could not import modules from 'src' directory.")
    print("Please ensure 'preprocess.py', 'diarization.py', 'transcribe.py', and 'utils.py' exist in a 'src' folder.")
    exit()

# --- Main Gradio Application Logic ---

def process_audio_pipeline(audio_path, num_speakers, *speaker_values, progress=gr.Progress(track_tqdm=True)):
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
        num_speakers_int = int(num_speakers) if num_speakers else 0
        diarization_segments, overlap_regions = diarization.diarize_audio(wav_path, num_speakers=num_speakers_int)
        
        if not diarization_segments:
            raise gr.Error("Speaker diarization failed or found no speakers.")

        # 3. Use diarization_segments (already a list of dicts) for transcription
        seg_dicts = diarization_segments
        # No need to convert from list-of-lists; diarization.py returns list of dicts

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

        # 5. Extract transcript segments and unique speakers
        progress(0.8, desc="Processing transcript...")
        transcript_segments = results['segments']
        unique_speakers = utils.extract_unique_speakers(transcript_segments)
        
        # 6. Create speaker mapping from user input
        speaker_names_dict = {}
        for i, name in enumerate(speaker_values):
            if name and name.strip():  # Only add non-empty names
                speaker_names_dict[f"SPEAKER_{i:02d}"] = name.strip()
        
        speaker_mapping = utils.create_speaker_mapping(speaker_names_dict, unique_speakers)
        
        # 7. Post-process transcript
        progress(0.9, desc="Formatting transcript...")
        processed_transcript = utils.post_process_transcript(transcript_segments, speaker_mapping)

        # 8. Create a downloadable transcript file
        transcript_filepath = os.path.join(temp_dir, "transcript.txt")
        with open(transcript_filepath, "w", encoding="utf-8") as f:
            f.write(processed_transcript)

        # 9. Final update to the UI
        progress(1.0, desc="Complete!")
        return wav_path, "Status: Transcription Complete!", processed_transcript, gr.File(value=transcript_filepath, visible=True)

    except Exception as e:
        # Clean up the temp directory on any failure
        shutil.rmtree(temp_dir)
        raise gr.Error(str(e))
    finally:
        # Ensure cleanup happens even on success, though transcript file needs to persist
        # Gradio handles the temp file created by gr.File, so we just log this.
        # The main temp_dir will be cleaned by the OS eventually.
        pass





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
            
            num_speakers_input = gr.Number(
                label="Number of Speakers (Optional)", 
                value=2, 
                info="Leave at 0 for automatic detection.",
                precision=0 # Ensures it's an integer
            )
            
            # Speaker name inputs
            gr.Markdown("### Speaker Name Assignment")
            gr.Markdown("Assign names to speakers (leave empty to keep default speaker labels):")
            
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

    # Create speaker input fields
    speaker_inputs = []
    for i in range(4):  # Create 4 speaker input fields by default
        speaker_inputs.append(
            gr.Textbox(
                label=f"Name for SPEAKER_{i:02d}",
                placeholder=f"Enter name for SPEAKER_{i:02d} (or leave empty for default)",
                value=""
            )
        )

    # Function to collect speaker names from inputs
    def collect_speaker_names(*speaker_values):
        speaker_names = {}
        for i, name in enumerate(speaker_values):
            if name and name.strip():  # Only add non-empty names
                speaker_names[f"SPEAKER_{i:02d}"] = name.strip()
        return speaker_names

    # Connect the button to the processing function
    submit_btn.click(
        fn=process_audio_pipeline,
        inputs=[audio_input, num_speakers_input] + speaker_inputs,
        outputs=[
            processed_audio_output,
            status_output,
            transcript_output,
            download_output
        ]
    )

if __name__ == "__main__":
    demo.launch(debug=True, share=True)