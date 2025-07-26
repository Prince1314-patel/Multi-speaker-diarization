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
    
    Args:
        diarization (Annotation): The initial diarization result from pyannote.
        overlap_regions (List[tuple]): A list of (start, end) tuples for overlap regions.
        
    Returns:
        List[Dict[str, any]]: A list of segments with overlaps resolved.
    """
    # Create a timeline of all original speaker turns
    original_timeline = diarization.get_timeline()
    
    # This will hold the final, processed segments
    resolved_segments = []

    # Process non-overlapping parts first
    # FIXED LOGIC: Correctly create a Timeline for gaps calculation
    overlap_timeline = Timeline([Segment(start, end) for start, end in overlap_regions])
    non_overlap_timeline = original_timeline.gaps(overlap_timeline)
    
    for segment in non_overlap_timeline:
        # Get speakers active in this non-overlapping segment
        speakers = diarization.crop(segment).labels()
        if speakers:
            # In non-overlap, there should ideally be only one speaker
            resolved_segments.append({
                'speaker': speakers[0],
                'start': segment.start,
                'end': segment.end,
                'is_overlap': False,
                'overlap_with': []
            })
            
    # Now, process overlapping parts
    for start, end in overlap_regions:
        overlap_segment = Segment(start, end)
        
        # Find all speakers active during this specific overlap segment
        speakers_in_overlap = diarization.crop(overlap_segment).labels()
        
        if len(speakers_in_overlap) > 1:
            # For each speaker in the overlap, create a distinct segment
            for speaker in speakers_in_overlap:
                resolved_segments.append({
                    'speaker': speaker,
                    'start': start,
                    'end': end,
                    'is_overlap': True,
                    'overlap_with': [s for s in speakers_in_overlap if s != speaker]
                })

    # Sort the final segments by start time
    resolved_segments.sort(key=lambda x: x['start'])
    return resolved_segments


def diarize_audio(audio_path: str) -> List[Dict[str, any]]:
    """
    Performs speaker diarization with active overlap resolution.

    Args:
        audio_path (str): Path to the preprocessed WAV audio file.

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
        diarization = diarization_pipeline(audio_path)
        logging.info("Initial diarization complete.")

        if torch.cuda.is_available():
            empty_cache()

        # 2. Detect overlap regions
        logging.info("Loading pyannote/overlapped-speech-detection pipeline...")
        overlap_detector = OverlapDetector(use_cuda=torch.cuda.is_available())
        if not overlap_detector.initialize(auth_token=hf_token):
             # Fallback to original diarization if overlap detector fails
            logging.warning("Overlap detector failed to initialize. Returning standard diarization.")
            return [
                {"speaker": speaker, "start": turn.start, "end": turn.end, "is_overlap": False, "overlap_with": []}
                for turn, _, speaker in diarization.itertracks(yield_label=True)
            ]
        
        overlap_regions = overlap_detector.detect_overlaps(audio_path)
        logging.info(f"Detected {len(overlap_regions)} overlap regions.")
        
        if not overlap_regions:
            logging.info("No overlaps detected. Returning standard diarization.")
            return [
                {"speaker": speaker, "start": turn.start, "end": turn.end, "is_overlap": False, "overlap_with": []}
                for turn, _, speaker in diarization.itertracks(yield_label=True)
            ]

        # 3. Resolve overlaps and create the final segment list
        final_segments = resolve_overlaps(diarization, overlap_regions)

        num_overlaps = sum(1 for seg in final_segments if seg['is_overlap'])
        logging.info(f"Diarization complete. Found {len(final_segments)} segments, "
                     f"with overlap resolution creating segments for {num_overlaps} speaker instances in overlaps.")

        return final_segments

    except Exception as e:
        logging.error(f"Error during diarization: {e}", exc_info=True)
        return []