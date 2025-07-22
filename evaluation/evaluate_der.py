import argparse
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.core import Annotation


def load_rttm(file_path):
    """Load an RTTM file as a pyannote.core.Annotation object.

    Args:
        file_path (str): Path to the RTTM file.

    Returns:
        Annotation: Loaded annotation.
    """
    annotation = Annotation()
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('SPEAKER'):
                parts = line.strip().split()
                start = float(parts[3])
                duration = float(parts[4])
                end = start + duration
                speaker = parts[7]
                annotation[(start, end)] = speaker
    return annotation


def main():
    """Compute DER between reference and hypothesis RTTM files."""
    parser = argparse.ArgumentParser(description="Compute Diarization Error Rate (DER)")
    parser.add_argument('--reference', required=True, help='Path to reference RTTM file')
    parser.add_argument('--hypothesis', required=True, help='Path to hypothesis RTTM file')
    args = parser.parse_args()

    reference = load_rttm(args.reference)
    hypothesis = load_rttm(args.hypothesis)

    metric = DiarizationErrorRate()
    der = metric(reference, hypothesis)
    detail = metric.compute_components(reference, hypothesis)

    print(f"DER: {der:.2%}")
    print("Breakdown:")
    print(f"  Missed speech: {detail['missed speech']:.2%}")
    print(f"  False alarm:   {detail['false alarm']:.2%}")
    print(f"  Confusion:     {detail['confusion']:.2%}")

if __name__ == "__main__":
    main() 