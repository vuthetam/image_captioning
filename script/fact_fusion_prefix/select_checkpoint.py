"""Select the epoch with the highest validation CIDEr for test generation."""
import sys
from functools import partial
from pathlib import Path

import pandas as pd
from accelerate import Accelerator
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fact_fusion_prefix.dataset import RAGFeatureDataset, collate_fact_batch
from src.shared.facts.encoder import CachedCLIPFactEncoder
from src.fact_fusion_prefix.inference_rag import generate_captions
from src.fact_fusion_prefix.model import FactFusionPrefixCaptioner
from script.fact_fusion_prefix.train_save_epochs import build_model_config
from src.shared.checkpoint_selection import ValidationCider, load_epoch_weights, select_best_checkpoint
from src.shared.config import (
    SEED, VOCAB_PATH, MAX_FACT_PREFIX_LENGTH,
    BATCH_SIZE, BEAM_SIZE, BEST_CHECKPOINT_PATH, RUN_CHECKPOINT_DIR,
    FACT_TEXT_BATCH_SIZE, MAX_LENGTH, NUM_WORKERS,
    VAL_GROUPED_DF_PATH, VAL_I2T_PATH, VAL_VISUAL_FEATURES_PATH,
    TEXT_RAG_CAPTION_SORT_BY, TOP_K_CAPTIONS, VISUAL_ENCODER_MODEL,
)
from src.shared.utils import log_reproducibility, setup_reproducibility
from src.shared.vocabulary import Vocabulary


def main() -> None:
    setup_reproducibility(SEED)
    accelerator = Accelerator()
    log_reproducibility(accelerator, SEED)
    checkpoint_paths = list(RUN_CHECKPOINT_DIR.glob("epoch_*.pth"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No epoch checkpoints in {RUN_CHECKPOINT_DIR}.")
    if not VOCAB_PATH.is_file():
        raise FileNotFoundError("Run python script/fact_fusion_prefix/build_vocab.py before generating captions.")
    vocab = Vocabulary.load(VOCAB_PATH)
    if "<sep>" not in vocab.stoi or "<sep>" not in vocab.special_tokens:
        raise ValueError("Prefix vocabulary must contain the SEP special token.")
    val_df = pd.read_parquet(VAL_GROUPED_DF_PATH)
    dataset = RAGFeatureDataset(
        val_df, vocab, VAL_VISUAL_FEATURES_PATH, VAL_I2T_PATH,
        TOP_K_CAPTIONS, MAX_FACT_PREFIX_LENGTH, TEXT_RAG_CAPTION_SORT_BY,
    )
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=True, collate_fn=partial(collate_fact_batch, pad_idx=vocab.pad_idx()),
    )
    with accelerator.main_process_first():
        fact_encoder = CachedCLIPFactEncoder(VISUAL_ENCODER_MODEL, accelerator.device, FACT_TEXT_BATCH_SIZE)
    model_config = build_model_config(dataset.feature_shape[-1], fact_encoder.output_dim, vocab)
    model = FactFusionPrefixCaptioner(**model_config["model_kwargs"])
    model, loader = accelerator.prepare(model, loader)
    scorer = ValidationCider(val_df) if accelerator.is_main_process else None

    def evaluate_checkpoint(checkpoint_path):
        epoch = load_epoch_weights(checkpoint_path, accelerator.unwrap_model(model), vocab, model_config)
        accelerator.print(f"Generating validation captions for epoch {epoch:02d}")
        captions = generate_captions(
            model, loader, vocab, BEAM_SIZE, MAX_LENGTH, accelerator, fact_encoder,
            length_penalty=1.0, show_progress=True, include_cls_token=False,
        )
        score = scorer.compute(captions) if accelerator.is_main_process else 0.0
        return epoch, score

    settings = {
        "selection_metric": "CIDEr", "split": "val_grouped", "seed": SEED,
        "validation_path": str(VAL_GROUPED_DF_PATH), "num_images": len(val_df),
        "beam_size": BEAM_SIZE, "max_length": MAX_LENGTH, "length_penalty": 1.0,
        "caption_sort_by": TEXT_RAG_CAPTION_SORT_BY, "top_k": TOP_K_CAPTIONS,
    }
    scores_path = RUN_CHECKPOINT_DIR / "checkpoint_scores.json"
    select_best_checkpoint(
        checkpoint_paths, evaluate_checkpoint, BEST_CHECKPOINT_PATH, scores_path, settings, accelerator,
    )
    accelerator.print(f"Saved {BEST_CHECKPOINT_PATH} and {scores_path}")
    accelerator.wait_for_everyone()
    sys.stdout.flush()


if __name__ == "__main__":
    main()
