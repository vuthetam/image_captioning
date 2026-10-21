import sys
import os
from pathlib import Path

import pandas as pd
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.checkpoint import load_checkpoint, save_checkpoint
from src.shared.config import (
    BATCH_SIZE,
    BEST_CHECKPOINT_PATH,
    DMODEL,
    DROPOUT,
    LAST_CHECKPOINT_PATH,
    LEARNING_RATE,
    MAX_GRAD_NORM,
    MAX_LENGTH,
    NHEADS,
    NLAYERS,
    NUM_EPOCHS,
    NUM_WORKERS,
    TRAIN_DF_PATH,
    TRAIN_VISUAL_FEATURES_PATH,
    VAL_DF_PATH,
    VAL_VISUAL_FEATURES_PATH,
    VOCAB_PATH,
    WEIGHT_DECAY,
)
from src.baseline.dataset import FeatureCaptionDataset
from src.baseline.engine import evaluate_one_epoch, train_one_epoch
from src.baseline.models.baseline import BaselineCaptioner
from src.shared.utils import (
    create_seeded_generator,
    log_reproducibility,
    seed_worker,
    setup_reproducibility,
    trainable_parameters,
)
from src.shared.vocabulary import Vocabulary


def main() -> None:
    setup_reproducibility(42)
    accelerator = Accelerator()
    log_reproducibility(accelerator, 42)

    train_df = pd.read_parquet(TRAIN_DF_PATH)
    val_df = pd.read_parquet(VAL_DF_PATH)
    vocab = Vocabulary.load(VOCAB_PATH)

    train_dataset = FeatureCaptionDataset(
        train_df, vocab, TRAIN_VISUAL_FEATURES_PATH, max_length=MAX_LENGTH
    )
    val_dataset = FeatureCaptionDataset(
        val_df, vocab, VAL_VISUAL_FEATURES_PATH, max_length=MAX_LENGTH
    )

    train_generator = create_seeded_generator(42)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = BaselineCaptioner(
        vocab_size=len(vocab),
        d_model=DMODEL,
        nheads=NHEADS,
        nlayers=NLAYERS,
        dropout=DROPOUT,
        max_length=MAX_LENGTH,
        pad_idx=vocab.pad_idx(),
        visual_feature_dim=train_dataset.feature_shape[-1],
    )
    
    optimizer = torch.optim.AdamW(
        trainable_parameters(model), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    start_epoch = 0
    best_val_loss = float("inf")
    if LAST_CHECKPOINT_PATH.exists():
        start_epoch, _, best_val_loss = load_checkpoint(
            LAST_CHECKPOINT_PATH, model, optimizer, device=accelerator.device
        )
        accelerator.print(f"Resuming after epoch {start_epoch}.")

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    
    accelerator.print(
        f"train rows={len(train_dataset):,}; val rows={len(val_dataset):,}; "
        f"feature shape={train_dataset.feature_shape}; trainable params="
        f"{sum(parameter.numel() for parameter in trainable_parameters(model)):,}"
    )

    for epoch in range(start_epoch, NUM_EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, vocab.pad_idx(), accelerator, MAX_GRAD_NORM, True
        )
        val_loss = evaluate_one_epoch(model, val_loader, vocab.pad_idx(), accelerator, True)
        
        accelerator.print(
            f"[Epoch {epoch + 1:02d}/{NUM_EPOCHS}] "
            f"train loss = {train_loss:.4f} val loss = {val_loss:.4f}"
        )

        if accelerator.is_main_process:
            # 1. Kiểm tra và cập nhật kỷ lục trước
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                save_checkpoint(
                    BEST_CHECKPOINT_PATH, model, optimizer, epoch + 1, train_loss, best_val_loss, accelerator
                )
                
            # 2. Lưu LAST checkpoint với kỷ lục ĐÃ CẬP NHẬT (nếu có)
            save_checkpoint(
                LAST_CHECKPOINT_PATH, model, optimizer, epoch + 1, train_loss, best_val_loss, accelerator
            )
            
    accelerator.print("BASELINE TRAINING COMPLETE!")
    
    # Ép Kaggle (Jupyter Notebook) kết thúc Kernel mạnh tay để tránh bị treo (zombie processes)
    accelerator.wait_for_everyone()
    sys.stdout.flush()
    os._exit(0)

if __name__ == "__main__":
    main()

