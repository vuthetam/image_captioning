import json
import sys
import os
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
    DMODEL,
    DROPOUT,
    MAX_LENGTH,
    NHEADS,
    NLAYERS,
    NUM_WORKERS,
    PREDICTIONS_PATH,
    TEST_DF_PATH,
    TEST_VISUAL_FEATURES_PATH,
    TEST_I2I_PATH,
    TRAIN_VISUAL_FEATURES_PATH,
    VOCAB_PATH,
    TOP_K_RAG_IMAGES,
)
from src.image_rag.dataset import RAGFeatureDataset
from src.image_rag.inference_rag import generate_captions
from src.image_rag.models.rag import ImageRAGCaptioner
from src.shared.utils import log_reproducibility, setup_reproducibility
from src.shared.vocabulary import Vocabulary


def main() -> None:
    setup_reproducibility(42)
    accelerator = Accelerator()
    log_reproducibility(accelerator, 42)
    test_df = pd.read_parquet(TEST_DF_PATH)
    test_related_df = pd.read_parquet(TEST_I2I_PATH)
    vocab = Vocabulary.load(VOCAB_PATH)
    
    test_dataset = RAGFeatureDataset(
        df=test_df, 
        related_df=test_related_df,
        visual_features_path=TEST_VISUAL_FEATURES_PATH,
        rag_features_path=TRAIN_VISUAL_FEATURES_PATH,
        top_k=TOP_K_RAG_IMAGES
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = ImageRAGCaptioner(
        vocab_size=len(vocab),
        d_model=DMODEL,
        nheads=NHEADS,
        nlayers=NLAYERS,
        dropout=DROPOUT,
        max_length=MAX_LENGTH,
        pad_idx=vocab.pad_idx(),
        visual_feature_dim=test_dataset.feature_shape[-1],
    )
    if not BEST_CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {BEST_CHECKPOINT_PATH}")
        
    epoch, _, _ = load_checkpoint(BEST_CHECKPOINT_PATH, model)
    accelerator.print(f"Đã nạp epoch {epoch}")

    model, test_loader = accelerator.prepare(model, test_loader)
    caption_dict = generate_captions(
        model, test_loader, vocab, BEAM_SIZE, MAX_LENGTH, accelerator, show_progress=True, include_cls_token=False
    )

    if accelerator.is_main_process:
        predictions = [
            {"imgid": int(imgid), "caption": " ".join(caption_dict[imgid])}
            for imgid in test_df["imgid"]
        ]
        PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with PREDICTIONS_PATH.open("w", encoding="utf-8") as output_file:
            json.dump(predictions, output_file, ensure_ascii=False, indent=2)
        accelerator.print(f"Saved {len(predictions):,} captions to {PREDICTIONS_PATH}")
        
    # Ép Kaggle/Jupyter kết thúc tiến trình
    accelerator.wait_for_everyone()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()

