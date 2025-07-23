from typing import List, Tuple
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def diarize_audio_nemo(audio_path: str, config_path: str = None) -> List[Tuple[str, float, float]]:
    """
    Performs speaker diarization using NVIDIA NeMo.
    Args:
        audio_path (str): Path to the preprocessed WAV audio file.
        config_path (str): Path to the NeMo diarization config YAML. Defaults to 'src/diar_infer_general.yaml'.
    Returns:
        List[Tuple[str, float, float]]: List of (speaker_label, start_time, end_time) segments.
    """
    try:
        from nemo.collections.asr.models import NeuralDiarizer
        from omegaconf import OmegaConf
        import json

        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'diar_infer_general.yaml')
        # Prepare manifest
        manifest = [{
            "audio_filepath": audio_path,
            "offset": 0,
            "duration": None,
            "label": "infer",
            "text": "-",
            "num_speakers": None,
            "rttm_filepath": None,
            "uem_filepath": None
        }]
        manifest_path = os.path.join(os.path.dirname(audio_path), "nemo_manifest.json")
        with open(manifest_path, "w") as f:
            for entry in manifest:
                f.write(json.dumps(entry) + "\n")

        # Load config
        cfg = OmegaConf.load(config_path)
        cfg.diarizer.manifest_filepath = manifest_path
        cfg.diarizer.out_dir = os.path.join(os.path.dirname(audio_path), "nemo_diarization_output")

        # Run diarization
        diarizer = NeuralDiarizer(cfg=cfg)
        diarizer.diarize()

        # Parse RTTM output
        base = os.path.splitext(os.path.basename(audio_path))[0]
        rttm_path = os.path.join(cfg.diarizer.out_dir, f"{base}.rttm")
        segments = []
        if os.path.exists(rttm_path):
            with open(rttm_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 8:
                        start = float(parts[3])
                        dur = float(parts[4])
                        speaker = parts[7]
                        segments.append((speaker, start, start + dur))
        else:
            logging.error(f"NeMo diarization RTTM not found: {rttm_path}")
        logging.info(f"NeMo diarization complete. Found {len(segments)} segments.")
        return segments
    except Exception as e:
        logging.error(f"Error during NeMo diarization: {e}")
        return [] 