"""Train the standardized V1-style text-RAG captioner."""

import os
import sys
from pathlib import Path

import pandas as pd
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.checkpoint import load_checkpoint, save_checkpoint
from src.shared.scheduler import create_accelerated_warmup_cosine_scheduler, optimizer_steps
from src.shared.config import (
    BATCH_SIZE,
    BEST_CHECKPOINT_PATH,
    CTX_NLAYERS,
    DMODEL,
    DROPOUT,
    LAST_CHECKPOINT_PATH,
    LEARNING_RATE,
    MAX_CTX_LENGTH,
    MAX_GRAD_NORM,
    MAX_LENGTH,
    NHEADS,
    NLAYERS,
    NUM_EPOCHS,
    NUM_WORKERS,
    TEXT_RAG_CAPTION_SORT_BY,
    TOP_K_CAPTIONS,
    TRAIN_DF_PATH,
    TRAIN_I2T_PATH,
    TRAIN_VISUAL_FEATURES_PATH,
    VAL_DF_PATH,
    VAL_I2T_PATH,
    VAL_VISUAL_FEATURES_PATH,
    VOCAB_PATH,
    WEIGHT_DECAY,
    WARMUP_RATIO,
    WARMUP_START_FACTOR,
    MIN_LR_RATIO,
)
from src.shared.utils import (
    create_seeded_generator,
    log_reproducibility,
    seed_worker,
    setup_reproducibility,
    trainable_parameters,
)
from src.shared.vocabulary import Vocabulary
from src.text_rag.dataset import RAGFeatureCaptionDataset
from src.text_rag.engine_rag import evaluate_one_epoch, train_one_epoch
from src.text_rag.models.rag import TextRAGCaptioner


def main() -> None:
    setup_reproducibility(42)
    accelerator = Accelerator()
    log_reproducibility(accelerator, 42)

    train_df = pd.read_parquet(TRAIN_DF_PATH)
    val_df = pd.read_parquet(VAL_DF_PATH)
    vocab = Vocabulary.load(VOCAB_PATH)
    train_dataset = RAGFeatureCaptionDataset(
        train_df,
        vocab,
        TRAIN_VISUAL_FEATURES_PATH,
        TRAIN_I2T_PATH,
        MAX_LENGTH,
        MAX_CTX_LENGTH,
        TOP_K_CAPTIONS,
        caption_sort_by=TEXT_RAG_CAPTION_SORT_BY,
    )
    val_dataset = RAGFeatureCaptionDataset(
        val_df,
        vocab,
        VAL_VISUAL_FEATURES_PATH,
        VAL_I2T_PATH,
        MAX_LENGTH,
        MAX_CTX_LENGTH,
        TOP_K_CAPTIONS,
        caption_sort_by=TEXT_RAG_CAPTION_SORT_BY,
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

    model = TextRAGCaptioner(
        vocab_size=len(vocab),
        d_model=DMODEL,
        nheads=NHEADS,
        nlayers=NLAYERS,
        dropout=DROPOUT,
        max_length=MAX_LENGTH,
        pad_idx=vocab.pad_idx(),
        visual_feature_dim=train_dataset.feature_shape[-1],
        context_layers=CTX_NLAYERS,
        max_context_length=MAX_CTX_LENGTH,
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters(model),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # Count optimizer updates after sharding; the scheduler library counts the
    # inner ticks that AcceleratedScheduler advances across all processes.
    train_loader, val_loader = accelerator.prepare(train_loader, val_loader)
    lr_schedule = create_accelerated_warmup_cosine_scheduler(
        optimizer,
        total_updates=len(train_loader) * NUM_EPOCHS,
        num_processes=accelerator.num_processes,
        warmup_ratio=WARMUP_RATIO,
        warmup_start_factor=WARMUP_START_FACTOR,
        min_lr_ratio=MIN_LR_RATIO,
    )

    start_epoch = 0
    best_val_loss = float("inf")
    if LAST_CHECKPOINT_PATH.is_file():
        start_epoch, _, best_val_loss = load_checkpoint(
            LAST_CHECKPOINT_PATH, model, optimizer, scheduler=lr_schedule,
        )
        accelerator.print(
            f"Resuming after epoch {start_epoch}; updates={optimizer_steps(lr_schedule)}; "
            f"best_val_loss={best_val_loss:.4f}."
        )

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, lr_schedule)

    accelerator.print(
        f"LR schedule: linear warmup + cosine; total updates={lr_schedule.schedule_config['total_updates']}; "
        f"warmup updates={lr_schedule.schedule_config['warmup_updates']}; "
        f"peak LR={lr_schedule.base_lrs[0]:.3e}; "
        f"current LR={optimizer.param_groups[0]['lr']:.3e}; "
        f"batch/GPU={BATCH_SIZE}; global batch={BATCH_SIZE * accelerator.num_processes}; "
        f"processes={accelerator.num_processes}"
    )
    accelerator.print(
        f"train rows={len(train_dataset):,}; val rows={len(val_dataset):,}; "
        f"caption sort={TEXT_RAG_CAPTION_SORT_BY}; "
        f"feature shape={train_dataset.feature_shape}; trainable params="
        f"{sum(parameter.numel() for parameter in trainable_parameters(model)):,}"
    )

    for epoch in range(start_epoch, NUM_EPOCHS):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            vocab.pad_idx(),
            accelerator,
            MAX_GRAD_NORM,
            show_progress=True,
            scheduler=scheduler,
        )
        val_loss = evaluate_one_epoch(
            model,
            val_loader,
            vocab.pad_idx(),
            accelerator,
            show_progress=True,
        )
        accelerator.print(
            f"[Epoch {epoch + 1:02d}/{NUM_EPOCHS}] "
            f"train loss = {train_loss:.4f} val loss = {val_loss:.4f}; "
            f"LR={optimizer.param_groups[0]['lr']:.3e}; updates={optimizer_steps(lr_schedule)}"
        )

        if accelerator.is_main_process:
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                save_checkpoint(
                    BEST_CHECKPOINT_PATH, model, optimizer, epoch + 1, train_loss, best_val_loss, accelerator, scheduler=scheduler
                )
            save_checkpoint(
                LAST_CHECKPOINT_PATH, model, optimizer, epoch + 1, train_loss, best_val_loss, accelerator, scheduler=scheduler
            )

    accelerator.print("TRAINING COMPLETE Text RAG!")
    
    accelerator.wait_for_everyone()
    sys.stdout.flush()
    os._exit(0)

if __name__ == "__main__":
    main()
