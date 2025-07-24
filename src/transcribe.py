import whisperx
import logging
from typing import List, Dict
import torch
from pyannote.core import Segment, Annotation

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def transcribe_with_whisperx(
    audio_path: str,
    diarization: List[Dict],
    model_name: str = "large-v2",
    device: str = "cuda",
    batch_size: int = 16,
    compute_type: str = "float16",
    use_vad: bool = True
) -> List[Dict]:
    """
    Transcribes an audio file using WhisperX, aligns with diarization, and uses VAD for segmentation.

    Args:
        audio_path (str): Path to the audio file.
        diarization (List[Dict]): List of diarization segments with 'speaker', 'start', and 'end'.
        model_name (str): Name of the WhisperX model to use.
        device (str): Device to run the model on (e.g., "cuda", "cpu").
        batch_size (int): Batch size for transcription.
        compute_type (str): Compute type for the model (e.g., "float16", "int8").
        use_vad (bool): Whether to use Voice Activity Detection (VAD) for pre-segmentation.

    Returns:
        List[Dict]: A list of transcribed segments, with speaker and word-level timestamps.
    """
    try:
        # 1. Load the WhisperX model
        model = whisperx.load_model(model_name, device, compute_type=compute_type)

        # 2. Load the audio
        audio = whisperx.load_audio(audio_path)

        # 3. Transcribe the audio
        logging.info("Transcribing audio with WhisperX...")
        result = model.transcribe(audio, batch_size=batch_size)
        logging.info(f"Transcription result: {result}")

        # 4. Align the transcription with diarization
        logging.info("Aligning transcription with diarization...")
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        aligned_result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        logging.info(f"Alignment result: {aligned_result}")

        # ---- START OF FIX ----
        # Before assigning speakers, filter out segments that have no words.
        # This prevents crashing the 'assign_word_speakers' function on empty segments.
        segments_with_words = [
            segment for segment in aligned_result["segments"] if 'words' in segment and len(segment['words']) > 0
        ]
        if not segments_with_words:
            logging.warning("No segments with words found after alignment. Returning empty list.")
            return []
        
        aligned_result["segments"] = segments_with_words
        # ---- END OF FIX ----

        # Convert diarization list of dicts to pyannote.core.Annotation
        diarization_annotation = Annotation()
        for segment_info in diarization:
            diarization_annotation[Segment(segment_info['start'], segment_info['end'])] = segment_info['speaker']

        # 5. Assign words to speakers using the provided diarization
        # 5. Assign words to speakers using the provided diarization
        final_result = whisperx.assign_word_speakers(diarization_annotation, aligned_result)

        # Reformat the output to match the expected structure
        logging.info("Reformatting output...")
        transcribed_segments = []
        for segment in final_result["segments"]:
            if 'speaker' in segment and 'start' in segment and 'end' in segment and 'text' in segment:
                transcribed_segments.append({
                    "speaker": segment["speaker"],
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"].strip(),
                    "words": segment.get("words", [])
                })

        return transcribed_segments

    except Exception as e:
        import traceback
        logging.error(f"Error during WhisperX transcription: {e}")
        logging.error(traceback.format_exc())
        return []