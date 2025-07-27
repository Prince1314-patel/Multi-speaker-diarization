import whisperx
import logging
from typing import List, Dict
import torch
import pandas as pd
import traceback

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def transcribe_with_whisperx(
    audio_path: str,
    diarization: List[Dict],
    model_name: str = "large-v2",
    device: str = "cuda",
    batch_size: int = 16,
    compute_type: str = "float16"
) -> Dict: # <-- CORRECTED RETURN TYPE
    """
    Transcribes an audio file using WhisperX and aligns with diarization.

    Args:
        audio_path (str): Path to the audio file.
        diarization (List[Dict]): List of diarization segments with 'speaker', 'start', and 'end'.
        model_name (str): Name of the WhisperX model to use.
        device (str): Device to run the model on (e.g., "cuda", "cpu").
        batch_size (int): Batch size for transcription.
        compute_type (str): Compute type for the model (e.g., "float16", "int8").

    Returns:
        Dict: A dictionary containing a list of transcribed segments under the 'segments' key.
    """
    try:
        # 1. Load the WhisperX model
        logging.info(f"Loading whisperx model: {model_name}")
        model = whisperx.load_model(model_name, device, compute_type=compute_type)

        # 2. Load the audio
        audio = whisperx.load_audio(audio_path)

        # 3. Transcribe the audio
        logging.info("Transcribing audio with WhisperX...")
        result = model.transcribe(audio, batch_size=batch_size)
        
        # Check if transcription produced any segments
        if not result.get("segments"):
            logging.warning("Transcription produced no segments.")
            return {"segments": []}

        # 4. Align the transcription
        logging.info("Aligning transcription...")
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        aligned_result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

        # Filter out segments that have no words after alignment.
        segments_with_words = [
            segment for segment in aligned_result["segments"] if 'words' in segment and len(segment['words']) > 0
        ]
        if not segments_with_words:
            logging.warning("No segments with words found after alignment. Returning empty result.")
            return {"segments": []}
        
        aligned_result["segments"] = segments_with_words

        # 5. Convert diarization list to a pandas DataFrame
        if not diarization:
            logging.warning("Diarization data is empty. Cannot assign speakers.")
            # Return the aligned transcript without speaker labels
            return aligned_result
        
        diarize_df = pd.DataFrame(diarization)
        if not all(col in diarize_df.columns for col in ['speaker', 'start', 'end']):
            raise ValueError("Diarization DataFrame must contain 'speaker', 'start', and 'end' columns.")

        # 6. Assign words to speakers
        logging.info("Assigning speakers to words...")
        final_result = whisperx.assign_word_speakers(diarize_df, aligned_result)

        # ---- START OF FIX ----
        # The final_result is already in the correct dictionary format.
        # We will just clean up the text and ensure the structure is consistent.
        
        logging.info("Finalizing transcript format...")
        
        # The final_result from whisperx is the dictionary we need.
        # We can optionally iterate through it to clean up text, but we must return the dictionary itself.
        if "segments" in final_result and final_result["segments"]:
            for segment in final_result["segments"]:
                if 'text' in segment:
                    segment['text'] = segment['text'].strip()
        
        # Return the entire dictionary, which contains the 'segments' key.
        return final_result
        # ---- END OF FIX ----

    except Exception as e:
        logging.error(f"Error during WhisperX transcription: {e}")
        logging.error(traceback.format_exc())
        # Return an empty structure that matches the expected output format
        return {"segments": []}

