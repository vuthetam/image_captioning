import sys
import os
from pathlib import Path

import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
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
    TRAIN_RELATED_IMAGES_PATH,
    VAL_DF_PATH,
    VAL_VISUAL_FEATURES_PATH,
    VAL_RELATED_IMAGES_PATH,
    VOCAB_PATH,
    WEIGHT_DECAY,
    TOP_K_RAG_IMAGES,
)
from src.v5.dataset import FeatureCaptionDatasetV5_RAG
from src.v5.engine_rag import evaluate_one_epoch_rag_v5, train_one_epoch_rag_v5
from src.v5.models.rag import RagCaptionerV5
from src.shared.utils import trainable_parameters
from src.shared.vocabulary import Vocabulary


def main() -> None:
    accelerator = Accelerator()
    set_seed(42)

    train_df = pd.read_parquet(TRAIN_DF_PATH)
    val_df = pd.read_parquet(VAL_DF_PATH)
    
    train_related_df = pd.read_parquet(TRAIN_RELATED_IMAGES_PATH)
    val_related_df = pd.read_parquet(VAL_RELATED_IMAGES_PATH)
    
    vocab = Vocabulary.load(VOCAB_PATH)

    train_dataset = FeatureCaptionDatasetV5_RAG(
        df=train_df,
        related_df=train_related_df,
        vocab=vocab,
        visual_features_path=TRAIN_VISUAL_FEATURES_PATH,
        rag_features_path=TRAIN_VISUAL_FEATURES_PATH,
        max_length=MAX_LENGTH,
        top_k=TOP_K_RAG_IMAGES
    )
    val_dataset = FeatureCaptionDatasetV5_RAG(
        df=val_df,
        related_df=val_related_df,
        vocab=vocab,
        visual_features_path=VAL_VISUAL_FEATURES_PATH,
        rag_features_path=TRAIN_VISUAL_FEATURES_PATH,
        max_length=MAX_LENGTH,
        top_k=TOP_K_RAG_IMAGES
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = RagCaptionerV5(
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
        train_loss = train_one_epoch_rag_v5(
            model, train_loader, optimizer, vocab.pad_idx(), accelerator, MAX_GRAD_NORM, show_progress=True, include_cls_token=False
        )
        val_loss = evaluate_one_epoch_rag_v5(
            model, val_loader, vocab.pad_idx(), accelerator, show_progress=True, include_cls_token=False
        )
        
        accelerator.print(
            f"[Epoch {epoch + 1:02d}/{NUM_EPOCHS}] "
            f"train loss = {train_loss:.4f} val loss = {val_loss:.4f}"
        )

        if accelerator.is_main_process:
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                save_checkpoint(
                    BEST_CHECKPOINT_PATH, model, optimizer, epoch + 1, train_loss, best_val_loss, accelerator
                )
                
            save_checkpoint(
                LAST_CHECKPOINT_PATH, model, optimizer, epoch + 1, train_loss, best_val_loss, accelerator
            )
            
    accelerator.print("TRAINING COMPLETE V5 RAG!")
    
    accelerator.wait_for_everyone()
    sys.stdout.flush()
    os._exit(0)

if __name__ == "__main__":
    main()
