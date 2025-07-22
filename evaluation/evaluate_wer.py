import argparse
from jiwer import wer, process_words


def read_text(file_path):
    """Read a text file and return its content as a string.

    Args:
        file_path (str): Path to the text file.

    Returns:
        str: File content.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def main():
    """Compute WER between reference and hypothesis transcripts."""
    parser = argparse.ArgumentParser(description="Compute Word Error Rate (WER)")
    parser.add_argument('--reference', required=True, help='Path to reference transcript (txt)')
    parser.add_argument('--hypothesis', required=True, help='Path to hypothesis transcript (txt)')
    args = parser.parse_args()

    ref = read_text(args.reference)
    hyp = read_text(args.hypothesis)

    error = process_words(ref, hyp)
    print(f"WER: {wer(ref, hyp):.2%}")
    print("Breakdown:")
    print(f"  Substitutions: {error.substitutions}")
    print(f"  Insertions:    {error.insertions}")
    print(f"  Deletions:     {error.deletions}")
    print(f"  Hits:          {error.hits}")

if __name__ == "__main__":
    main() 