"""Train from scratch and save each epoch for offline validation-CIDEr selection."""
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
from src.fact_fusion_token_bias.engine_rag import train_one_epoch
from src.fact_fusion_token_bias.model import FactFusionTokenBiasCaptioner
from src.shared.checkpoint import save_epoch_checkpoint
from src.shared.config import (
    SEED,
    BATCH_SIZE, RUN_CHECKPOINT_DIR, RETRIEVAL_OPENCLIP_MODEL_ID, RAG_CONTEXTS_SUBDIR, DMODEL, DROPOUT, FACT_FUSION_NLAYERS,
    FACT_TEXT_BATCH_SIZE, LEARNING_RATE, MAX_GRAD_NORM,
    MAX_LENGTH, MIN_LR_RATIO, NHEADS, NLAYERS, NUM_EPOCHS, NUM_WORKERS, RUN_MODE,
    TEXT_RAG_CAPTION_SORT_BY, TOP_K_CAPTIONS, TRAIN_DF_PATH, TRAIN_I2T_PATH,
    TRAIN_VISUAL_FEATURES_PATH,
    VISUAL_ENCODER_MODEL, VOCAB_PATH, WARMUP_RATIO, WARMUP_START_FACTOR, WEIGHT_DECAY,
)
from src.shared.scheduler import create_accelerated_warmup_cosine_scheduler
from src.shared.utils import (
    create_seeded_generator, log_reproducibility, seed_worker,
    setup_reproducibility, trainable_parameters,
)
from src.shared.vocabulary import Vocabulary


def build_model_config(visual_feature_dim, fact_feature_dim, vocab):
    model_kwargs = dict(
        vocab_size=len(vocab), d_model=DMODEL, nheads=NHEADS, nlayers=NLAYERS,
        dropout=DROPOUT, max_length=MAX_LENGTH, pad_idx=vocab.pad_idx(),
        visual_feature_dim=visual_feature_dim,
        fact_feature_dim=fact_feature_dim, fusion_layers=FACT_FUSION_NLAYERS,
        initial_alpha=0.3,
    )
    return {
        "pipeline": "fact_fusion_token_bias", "seed": SEED, "model_kwargs": model_kwargs,
        "visual_encoder_model": VISUAL_ENCODER_MODEL,
        "retrieval_model": RETRIEVAL_OPENCLIP_MODEL_ID,
        "rag_contexts_subdir": RAG_CONTEXTS_SUBDIR,
        "top_k": TOP_K_CAPTIONS, "caption_sort_by": TEXT_RAG_CAPTION_SORT_BY,
        "include_cls_token": False,
    }


def main() -> None:
    setup_reproducibility(SEED)
    accelerator = Accelerator()
    log_reproducibility(accelerator, SEED)
    if any(RUN_CHECKPOINT_DIR.glob("epoch_*.pth")):
        raise FileExistsError(f"Epoch checkpoints already exist in {RUN_CHECKPOINT_DIR}. Use a new RUN_MODE.")
    vocab = Vocabulary.load(VOCAB_PATH)
    train_dataset = RAGFeatureCaptionDataset(
        pd.read_parquet(TRAIN_DF_PATH), vocab, TRAIN_VISUAL_FEATURES_PATH, TRAIN_I2T_PATH,
        MAX_LENGTH, TOP_K_CAPTIONS, TEXT_RAG_CAPTION_SORT_BY,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        pin_memory=True, drop_last=True, collate_fn=collate_fact_batch,
        worker_init_fn=seed_worker, generator=create_seeded_generator(SEED),
    )
    # Each rank owns its frozen text tower and lazy CPU cache.
    with accelerator.main_process_first():
        fact_encoder = CachedCLIPFactEncoder(VISUAL_ENCODER_MODEL, accelerator.device, FACT_TEXT_BATCH_SIZE)

    model_config = build_model_config(train_dataset.feature_shape[-1], fact_encoder.output_dim, vocab)
    model = FactFusionTokenBiasCaptioner(**model_config["model_kwargs"])
    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_loader = accelerator.prepare(train_loader)
    lr_schedule = create_accelerated_warmup_cosine_scheduler(
        optimizer, total_updates=len(train_loader) * NUM_EPOCHS, num_processes=accelerator.num_processes,
        warmup_ratio=WARMUP_RATIO, warmup_start_factor=WARMUP_START_FACTOR, min_lr_ratio=MIN_LR_RATIO,
    )
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, lr_schedule)
    accelerator.print(
        f"Fact Fusion Token Bias; run mode={RUN_MODE}; sort={TEXT_RAG_CAPTION_SORT_BY}; top-k={TOP_K_CAPTIONS}; "
        f"batch/GPU={BATCH_SIZE}; global batch={BATCH_SIZE * accelerator.num_processes}; "
        f"total params={sum(p.numel() for p in model.parameters()):,}; "
        f"trainable params={sum(p.numel() for p in trainable_parameters(model)):,}"
    )
    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, vocab, accelerator, fact_encoder,
            MAX_GRAD_NORM, show_progress=True, scheduler=scheduler,
            include_cls_token=False,
        )
        accelerator.print(
            f"[Epoch {epoch + 1:02d}/{NUM_EPOCHS}] train loss={train_loss:.4f}; "
            f"LR={optimizer.param_groups[0]['lr']:.3e}; "
            f"alpha={accelerator.unwrap_model(model).decoder.alpha.item():.4f}"
        )
        if accelerator.is_main_process:
            checkpoint_path = RUN_CHECKPOINT_DIR / f"epoch_{epoch + 1:02d}.pth"
            save_epoch_checkpoint(
                checkpoint_path, model, epoch + 1, train_loss, model_config, vocab, SEED, accelerator,
            )
            accelerator.print(f"Saved {checkpoint_path}")
        accelerator.wait_for_everyone()
    accelerator.print("TRAINING COMPLETE Fact Fusion Token Bias!")
    accelerator.wait_for_everyone()
    sys.stdout.flush()


if __name__ == "__main__":
    main()
