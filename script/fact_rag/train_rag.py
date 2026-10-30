"""Train a captioner with image-patch queries and retrieved CLIP fact keys/values."""
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

from src.shared.facts.dataset import RAGFeatureCaptionDataset, collate_fact_batch
from src.shared.facts.encoder import CachedCLIPFactEncoder
from src.fact_rag.engine_rag import evaluate_one_epoch, train_one_epoch
from src.fact_rag.models import FactRAGCaptioner
from src.shared.checkpoint import load_checkpoint, save_checkpoint
from src.shared.config import (
    BATCH_SIZE, BEST_CHECKPOINT_PATH, DMODEL, DROPOUT, FACT_FUSION_NLAYERS,
    FACT_TEXT_BATCH_SIZE, LAST_CHECKPOINT_PATH, LEARNING_RATE, MAX_GRAD_NORM,
    MAX_LENGTH, MIN_LR_RATIO, NHEADS, NLAYERS, NUM_EPOCHS, NUM_WORKERS, RUN_MODE,
    TEXT_RAG_CAPTION_SORT_BY, TOP_K_CAPTIONS, TRAIN_DF_PATH, TRAIN_I2T_PATH,
    TRAIN_VISUAL_FEATURES_PATH, VAL_DF_PATH, VAL_I2T_PATH, VAL_VISUAL_FEATURES_PATH,
    VISUAL_ENCODER_MODEL, VOCAB_PATH, WARMUP_RATIO, WARMUP_START_FACTOR, WEIGHT_DECAY,
)
from src.shared.scheduler import create_accelerated_warmup_cosine_scheduler, optimizer_steps
from src.shared.utils import (
    create_seeded_generator, log_reproducibility, seed_worker,
    setup_reproducibility, trainable_parameters,
)
from src.shared.vocabulary import Vocabulary


def main() -> None:
    setup_reproducibility(42)
    accelerator = Accelerator()
    log_reproducibility(accelerator, 42)
    vocab = Vocabulary.load(VOCAB_PATH)
    train_dataset = RAGFeatureCaptionDataset(
        pd.read_parquet(TRAIN_DF_PATH), vocab, TRAIN_VISUAL_FEATURES_PATH, TRAIN_I2T_PATH,
        MAX_LENGTH, TOP_K_CAPTIONS, TEXT_RAG_CAPTION_SORT_BY,
    )
    val_dataset = RAGFeatureCaptionDataset(
        pd.read_parquet(VAL_DF_PATH), vocab, VAL_VISUAL_FEATURES_PATH, VAL_I2T_PATH,
        MAX_LENGTH, TOP_K_CAPTIONS, TEXT_RAG_CAPTION_SORT_BY,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        pin_memory=True, drop_last=True, collate_fn=collate_fact_batch,
        worker_init_fn=seed_worker, generator=create_seeded_generator(42),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=True, collate_fn=collate_fact_batch,
    )
    # Each rank owns a frozen text tower and CPU cache, shared by train and val.
    with accelerator.main_process_first():
        fact_encoder = CachedCLIPFactEncoder(VISUAL_ENCODER_MODEL, accelerator.device, FACT_TEXT_BATCH_SIZE)
        
    model = FactRAGCaptioner(
        vocab_size=len(vocab), d_model=DMODEL, nheads=NHEADS, nlayers=NLAYERS,
        dropout=DROPOUT, max_length=MAX_LENGTH, pad_idx=vocab.pad_idx(),
        visual_feature_dim=train_dataset.feature_shape[-1],
        fact_feature_dim=fact_encoder.output_dim, fusion_layers=FACT_FUSION_NLAYERS,
    )
    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_loader, val_loader = accelerator.prepare(train_loader, val_loader)
    lr_schedule = create_accelerated_warmup_cosine_scheduler(
        optimizer, total_updates=len(train_loader) * NUM_EPOCHS, num_processes=accelerator.num_processes,
        warmup_ratio=WARMUP_RATIO, warmup_start_factor=WARMUP_START_FACTOR, min_lr_ratio=MIN_LR_RATIO,
    )
    start_epoch, best_val_loss = 0, float("inf")
    if LAST_CHECKPOINT_PATH.is_file():
        start_epoch, _, best_val_loss = load_checkpoint(
            LAST_CHECKPOINT_PATH, model, optimizer, scheduler=lr_schedule,
        )
        accelerator.print(f"Resuming after epoch {start_epoch}; updates={optimizer_steps(lr_schedule)}")
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, lr_schedule)
    accelerator.print(
        f"Fact RAG; run mode={RUN_MODE}; CLIP Text={VISUAL_ENCODER_MODEL}; "
        f"fact dim={fact_encoder.output_dim}; sort={TEXT_RAG_CAPTION_SORT_BY}; "
        f"top K captions={TOP_K_CAPTIONS}; fusion layers={FACT_FUSION_NLAYERS}; "
        f"cache=lazy CPU RAM per process; feature shape={train_dataset.feature_shape}; "
        f"train rows={len(train_dataset):,}; val rows={len(val_dataset):,}; "
        f"batch/GPU={BATCH_SIZE}; global batch={BATCH_SIZE * accelerator.num_processes}; "
        f"trainable params={sum(p.numel() for p in trainable_parameters(model)):,}"
    )
    for epoch in range(start_epoch, NUM_EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, vocab.pad_idx(), accelerator, fact_encoder,
            MAX_GRAD_NORM, show_progress=True, scheduler=scheduler,
            include_cls_token=False,
        )
        val_loss = evaluate_one_epoch(
            model, val_loader, vocab.pad_idx(), accelerator, fact_encoder, show_progress=True,
            include_cls_token=False,
        )
        accelerator.print(
            f"[Epoch {epoch + 1:02d}/{NUM_EPOCHS}] train loss={train_loss:.4f}; val loss={val_loss:.4f}; "
            f"LR={optimizer.param_groups[0]['lr']:.3e}; updates={optimizer_steps(lr_schedule)}"
        )
        if accelerator.is_main_process:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    BEST_CHECKPOINT_PATH, model, optimizer, epoch + 1, train_loss,
                    best_val_loss, accelerator, scheduler=scheduler,
                )
            save_checkpoint(
                LAST_CHECKPOINT_PATH, model, optimizer, epoch + 1, train_loss,
                best_val_loss, accelerator, scheduler=scheduler,
            )
    accelerator.print("TRAINING COMPLETE Fact RAG!")
    accelerator.wait_for_everyone()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
