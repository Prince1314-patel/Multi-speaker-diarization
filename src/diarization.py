# diarization.py
from typing import List, Dict
import os
import logging
from pyannote.audio import Pipeline as DiarizationPipeline
from pyannote.core import Annotation, Segment, Timeline
import torch
from src.overlap import OverlapDetector
from torch.cuda import empty_cache

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def resolve_overlaps(diarization: Annotation, overlap_regions: List[tuple]) -> List[Dict[str, any]]:
    """
    Resolves overlaps in a diarization by creating multi-speaker segments.
    """
    original_timeline = diarization.get_timeline()
    resolved_segments = []

    if overlap_regions:
        overlap_timeline = Timeline([Segment(start, end) for start, end in overlap_regions])
        non_overlap_timeline = original_timeline.gaps(overlap_timeline)
    else:
        non_overlap_timeline = original_timeline

    for segment in non_overlap_timeline:
        speakers = diarization.crop(segment).labels()
        if speakers:
            resolved_segments.append({
                'speaker': speakers[0],
                'start': segment.start,
                'end': segment.end,
                'is_overlap': False,
                'overlap_with': []
            })
            
    for start, end in overlap_regions:
        overlap_segment = Segment(start, end)
        speakers_in_overlap = diarization.crop(overlap_segment).labels()
        
        if len(speakers_in_overlap) > 1:
            for speaker in speakers_in_overlap:
                resolved_segments.append({
                    'speaker': speaker,
                    'start': start,
                    'end': end,
                    'is_overlap': True,
                    'overlap_with': [s for s in speakers_in_overlap if s != speaker]
                })

    resolved_segments.sort(key=lambda x: x['start'])
    return resolved_segments

# --- MODIFIED FUNCTION ---
def diarize_audio(audio_path: str, num_speakers: int = 0) -> List[Dict[str, any]]:
    """
    Performs speaker diarization with active overlap resolution.

    Args:
        audio_path (str): Path to the preprocessed WAV audio file.
        num_speakers (int): The number of speakers to detect. If 0, the model detects automatically.

    Returns:
        List[Dict[str, any]]: List of segments with resolved overlaps.
    """
    try:
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        if not hf_token:
            logging.error("Hugging Face token not found in environment variable 'HUGGINGFACE_TOKEN'.")
            return []
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")

        # 1. Run standard diarization
        logging.info("Loading pyannote/speaker-diarization-3.1 pipeline...")
        diarization_pipeline = DiarizationPipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        ).to(device)

        # --- START OF FIX ---
        # Use the num_speakers parameter if provided by the user
        pipeline_params = {}
        if num_speakers > 0:
            pipeline_params["num_speakers"] = num_speakers
            logging.info(f"Diarizing with a fixed number of speakers: {num_speakers}")
        else:
            logging.info("Diarizing with automatic speaker number detection.")
        
        diarization = diarization_pipeline(audio_path, **pipeline_params)
        # --- END OF FIX ---

        logging.info("Initial diarization complete.")

        if torch.cuda.is_available():
            empty_cache()

        # 2. Detect overlap regions
        logging.info("Loading pyannote/overlapped-speech-detection pipeline...")
        overlap_detector = OverlapDetector(use_cuda=torch.cuda.is_available())
        if not overlap_detector.initialize(auth_token=hf_token):
            logging.warning("Overlap detector failed to initialize. Returning standard diarization.")
            return [
                {"speaker": speaker, "start": turn.start, "end": turn.end, "is_overlap": False, "overlap_with": []}
                for turn, _, speaker in diarization.itertracks(yield_label=True)
            ]
        
        overlap_regions = overlap_detector.detect_overlaps(audio_path)
        logging.info(f"Detected {len(overlap_regions)} overlap regions.")
        
        # 3. Resolve overlaps and create the final segment list
        if overlap_regions:
            final_segments = resolve_overlaps(diarization, overlap_regions)
        else:
            logging.info("No overlaps detected. Formatting standard diarization output.")
            final_segments = [
                {"speaker": speaker, "start": turn.start, "end": turn.end, "is_overlap": False, "overlap_with": []}
                for turn, _, speaker in diarization.itertracks(yield_label=True)
            ]


        num_overlaps = sum(1 for seg in final_segments if seg.get('is_overlap'))
        logging.info(f"Diarization complete. Found {len(final_segments)} segments, "
                     f"with {num_overlaps} speaker instances in overlaps.")

        return final_segments

    except Exception as e:
        logging.error(f"Error during diarization: {e}", exc_info=True)
        return []
