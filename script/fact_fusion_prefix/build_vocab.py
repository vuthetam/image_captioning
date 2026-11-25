"""Create the separate prefix vocabulary before training."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.config import VOCAB_PATH
from src.shared.vocabulary import Vocabulary


def build_prefix_vocabulary(vocab_path: Path, output_path: Path) -> Vocabulary:
    if vocab_path.resolve() == output_path.resolve():
        raise ValueError("Prefix vocabulary must use a separate path from the base vocabulary.")
    vocab = Vocabulary.load(vocab_path)
    if "<sep>" not in vocab.stoi:
        vocab.stoi["<sep>"] = len(vocab.itos)
        vocab.itos.append("<sep>")
    if "<sep>" not in vocab.special_tokens:
        vocab.special_tokens.append("<sep>")

    if output_path.exists():
        saved = Vocabulary.load(output_path)
        if saved.to_state() != vocab.to_state():
            raise ValueError("Saved prefix vocabulary differs from the extended base vocabulary.")
        return saved
    vocab.save(output_path)
    return vocab


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-path", type=Path, default=VOCAB_PATH, help="Original decoder vocabulary.")
    parser.add_argument("--output-path", type=Path, required=True, help="Separate prefix vocabulary.")
    args = parser.parse_args(argv)

    vocab = build_prefix_vocabulary(args.vocab_path, args.output_path)
    print(f"Prefix vocabulary ready: {args.output_path}; tokens={len(vocab)}; sep_id={vocab.sep_idx()}")


if __name__ == "__main__":
    main()
