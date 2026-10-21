"""Generate captions with independently retrieved image and text contexts."""

import argparse
import os
import json
import sys
from pathlib import Path

import pandas as pd
from accelerate import Accelerator
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.checkpoint import load_checkpoint
from src.shared.config import (
    BATCH_SIZE,
    BEAM_SIZE,
    BEST_CHECKPOINT_PATH,
    CTX_NLAYERS,
    DMODEL,
    DROPOUT,
    MAX_CTX_LENGTH,
    MAX_LENGTH,
    NHEADS,
    NLAYERS,
    NUM_WORKERS,
    PREDICTIONS_PATH,
    TEST_DF_PATH,
    TEST_I2T_PATH,
    TEST_I2I_PATH,
    TEST_VISUAL_FEATURES_PATH,
    RUN_MODE,
    TOP_K_CAPTIONS,
    TOP_K_RAG_IMAGES,
    TRAIN_VISUAL_FEATURES_PATH,
    VOCAB_PATH,
)
from src.shared.utils import log_reproducibility, setup_reproducibility
from src.shared.vocabulary import Vocabulary
from src.multimodal_rag.dataset import RAGFeatureDataset
from src.multimodal_rag.inference_rag import generate_captions
from src.multimodal_rag.models import MultimodalRAGCaptioner, MultimodalRAGCaptionerNoInputImage


def main(no_input_image: bool = False) -> None:
    setup_reproducibility(42)
    accelerator = Accelerator()
    log_reproducibility(accelerator, 42)

    test_df = pd.read_parquet(TEST_DF_PATH)
    vocab = Vocabulary.load(VOCAB_PATH)
    test_dataset = RAGFeatureDataset(
        df=test_df,
        vocab=vocab,
        features_path=None if no_input_image else TEST_VISUAL_FEATURES_PATH,
        related_features_path=TRAIN_VISUAL_FEATURES_PATH,
        rag_contexts_path=TEST_I2T_PATH,
        related_images_path=TEST_I2I_PATH,
        max_context_length=MAX_CTX_LENGTH,
        top_k_captions=TOP_K_CAPTIONS,
        top_k_images=TOP_K_RAG_IMAGES,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    model_class = MultimodalRAGCaptionerNoInputImage if no_input_image else MultimodalRAGCaptioner
    model = model_class(
        vocab_size=len(vocab),
        d_model=DMODEL,
        nheads=NHEADS,
        nlayers=NLAYERS,
        dropout=DROPOUT,
        max_length=MAX_LENGTH,
        pad_idx=vocab.pad_idx(),
        visual_feature_dim=test_dataset.feature_shape[-1],
        context_layers=CTX_NLAYERS,
        max_context_length=MAX_CTX_LENGTH,
    )

    if not BEST_CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {BEST_CHECKPOINT_PATH}")
    epoch, _, best_val_loss = load_checkpoint(
        BEST_CHECKPOINT_PATH, model, device=accelerator.device
    )
    accelerator.print(f"model={model_class.__name__}; run mode={RUN_MODE}; loaded epoch {epoch}; best val loss: {best_val_loss}")

    model, test_loader = accelerator.prepare(model, test_loader)
    caption_dict = generate_captions(
        model,
        test_loader,
        vocab,
        BEAM_SIZE,
        MAX_LENGTH,
        accelerator,
        show_progress=True,
    )

    if accelerator.is_main_process:
        predictions = [
            {"imgid": int(imgid), "caption": " ".join(caption_dict[int(imgid)])}
            for imgid in test_df["imgid"]
        ]
        PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with PREDICTIONS_PATH.open("w", encoding="utf-8") as output_file:
            json.dump(predictions, output_file, ensure_ascii=False, indent=2)
        accelerator.print(
            f"Saved {len(predictions):,} captions to {PREDICTIONS_PATH}"
        )

    accelerator.wait_for_everyone()
    sys.stdout.flush()
    os._exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-input-image", action="store_true", help="Use only related-image CLS tokens and retrieved captions as decoder memory.")
    args = parser.parse_args()
    main(no_input_image=args.no_input_image)
