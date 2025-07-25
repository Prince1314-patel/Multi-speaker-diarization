"""
Overlap detection module using Pyannote's dedicated overlapped speech detection model.
This module provides functionality to detect regions of overlapped speech in audio files.
"""

from typing import List, Tuple
import os
from dotenv import load_dotenv
import logging
import torch
from pyannote.audio import Pipeline
from pyannote.core import Segment, Timeline

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

class OverlapDetector:
    def __init__(self, use_cuda: bool = True):
        """
        Initialize the overlap detector with Pyannote's overlapped speech detection model.
        
        Args:
            use_cuda (bool): Whether to use CUDA for GPU acceleration
        """
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        self.pipeline = None
        
    def initialize(self, auth_token: str) -> bool:
        """
        Initialize the Pyannote overlap detection pipeline.
        
        Args:
            auth_token (str): HuggingFace authentication token
            
        Returns:
            bool: True if initialization successful, False otherwise
        """
        auth_token = os.getenv("HUGGINGFACE_TOKEN")
        if not auth_token:
            logging.error("HUGGINGFACE_TOKEN environment variable not set")
            return False
        try:
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/overlapped-speech-detection",
                use_auth_token=auth_token,
            ).to(self.device)
            return True
        except Exception as e:
            logging.error(f"Failed to initialize overlap detector: {e}")
            return False
            
    def detect_overlaps(self, audio_path: str, min_duration_ms: float = 100) -> List[Tuple[float, float]]:
        """
        Detect regions of overlapped speech in the audio file.
        
        Args:
            audio_path (str): Path to the audio file
            min_duration_ms (float): Minimum duration in milliseconds for an overlap region
            
        Returns:
            List[Tuple[float, float]]: List of (start_time, end_time) tuples for overlap regions
        """
        if not self.pipeline:
            logging.error("Pipeline not initialized. Call initialize() first.")
            return []
            
        try:
            # Run overlap detection
            output = self.pipeline(audio_path)
            
            # Convert min_duration to seconds
            min_duration = min_duration_ms / 1000
            
            # Extract overlap regions
            overlap_regions = []
            for segment in output.get_timeline():
                # Filter out very short overlaps
                if segment.duration < min_duration:
                    continue
                overlap_regions.append((segment.start, segment.end))
            
            logging.info(f"Detected {len(overlap_regions)} overlap regions in {audio_path}")
            return overlap_regions
        except Exception as e:
            logging.error(f"Failed to detect overlaps in audio file: {e}")
            return []
            
    def merge_with_diarization(self, diarization_segments: List[Tuple[str, float, float]], 
                             overlap_regions: List[Tuple[float, float]]) -> List[Tuple[str, float, float, bool]]:
        """
        Merge diarization segments with detected overlap regions.
        
        Args:
            diarization_segments: List of (speaker_label, start_time, end_time) from diarization
            overlap_regions: List of (start_time, end_time) from overlap detection
            
        Returns:
            List[Tuple[str, float, float, bool]]: Enhanced segments with overlap flags
        """
        # Create a timeline of overlap regions
        overlap_timeline = Timeline()
        for start, end in overlap_regions:
            overlap_timeline.add(Segment(start, end))
        
        # Enhance diarization segments with overlap information
        enhanced_segments = []
        for speaker, start, end in diarization_segments:
            segment = Segment(start, end)
            # Check if this segment intersects with any overlap region
            is_overlap = any(segment.intersects(ovl) for ovl in overlap_timeline)
            enhanced_segments.append((speaker, start, end, is_overlap))
            
        return enhanced_segments
