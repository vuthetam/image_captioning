"""Offline validation-CIDEr selection shared by the fact captioning pipelines."""

import json
import math
import shutil
from pathlib import Path

import torch
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer


class ValidationCider:
    def __init__(self, dataframe):
        if dataframe.empty or dataframe["imgid"].duplicated().any():
            raise ValueError("Grouped validation must contain one row per image.")
        references = {}
        for imgid, captions in dataframe[["imgid", "all_raws"]].itertuples(index=False, name=None):
            captions = list(captions)
            if not captions or any(not isinstance(caption, str) or not caption.strip() for caption in captions):
                raise ValueError(f"Missing reference captions for imgid={imgid}.")
            references[int(imgid)] = [{"caption": caption} for caption in captions]
        self.tokenizer = PTBTokenizer()
        self.references = self.tokenizer.tokenize(references)

    def compute(self, captions):
        if set(captions) != set(self.references):
            raise ValueError("Generated captions must cover every validation image exactly once.")
        predictions = {imgid: [{"caption": " ".join(tokens)}] for imgid, tokens in captions.items()}
        predictions = self.tokenizer.tokenize(predictions)
        score, _ = Cider().compute_score(self.references, predictions)
        if not math.isfinite(score):
            raise ValueError("Validation CIDEr must be finite.")
        return float(score)


def load_epoch_weights(path, model, vocab, model_config):
    """Reject configuration or vocabulary drift even when tensor shapes still match."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["model_config"] != model_config:
        raise ValueError(f"Model/retrieval configuration differs from checkpoint: {path}")
    if checkpoint["vocab_state"] != vocab.to_state():
        raise ValueError(f"Vocabulary differs from checkpoint: {path}")
    if checkpoint["epoch"] != int(Path(path).stem.removeprefix("epoch_")):
        raise ValueError(f"Epoch metadata does not match checkpoint filename: {path}")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return checkpoint["epoch"]


def select_best_checkpoint(checkpoint_paths, evaluate_checkpoint, best_path, scores_path, settings, accelerator):
    """Evaluate on every rank, publish one cumulative report and the best weights on rank zero."""
    checkpoint_paths = sorted(checkpoint_paths, key=lambda path: int(path.stem.removeprefix("epoch_")))
    if not checkpoint_paths:
        raise FileNotFoundError("No epoch_*.pth checkpoints found. Run train_save_epochs.py first.")
    best_path, scores_path = Path(best_path), Path(scores_path)
    report = dict(settings, epochs=[], best_epoch=None, best_CIDEr=None, best_checkpoint=str(best_path))
    best_score = float("-inf")
    for checkpoint_path in checkpoint_paths:
        epoch, score = evaluate_checkpoint(checkpoint_path)
        # Only the main process computes CIDEr after distributed caption gathering.
        score_tensor = torch.tensor(score if accelerator.is_main_process else 0.0,
                                    dtype=torch.float64, device=accelerator.device)
        score = accelerator.reduce(score_tensor, reduction="sum").item()
        if not math.isfinite(score):
            raise ValueError(f"Invalid validation CIDEr for epoch {epoch}.")
        report["epochs"].append({"epoch": epoch, "checkpoint": checkpoint_path.name, "CIDEr": score})
        if score > best_score:
            best_score = score
            report["best_epoch"], report["best_CIDEr"] = epoch, score
            if accelerator.is_main_process:
                best_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(checkpoint_path, best_path)
        if accelerator.is_main_process:
            scores_path.parent.mkdir(parents=True, exist_ok=True)
            scores_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        accelerator.print(f"Epoch {epoch:02d}; validation CIDEr={score * 100:.3f}; best epoch={report['best_epoch']}")
        accelerator.wait_for_everyone()
    return report
