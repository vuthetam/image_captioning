"""Generate test captions from fused image-fact tokens and direct fact memory."""
import json
import sys
from pathlib import Path

import pandas as pd
from accelerate import Accelerator
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.facts.dataset import RAGFeatureDataset, collate_fact_batch
from src.shared.facts.encoder import CachedCLIPFactEncoder
from src.fact_memory_rag.inference_rag import generate_captions
from src.fact_memory_rag.models import FactMemoryRAGCaptioner
from src.shared.checkpoint import load_checkpoint
from src.shared.config import (
    BATCH_SIZE, BEAM_SIZE, BEST_CHECKPOINT_PATH, DMODEL, DROPOUT, FACT_FUSION_NLAYERS,
    FACT_TEXT_BATCH_SIZE, MAX_LENGTH, NHEADS, NLAYERS, NUM_WORKERS, PREDICTIONS_PATH,
    RUN_MODE, TEST_DF_PATH, TEST_I2T_PATH, TEST_VISUAL_FEATURES_PATH,
    TEXT_RAG_CAPTION_SORT_BY, TOP_K_CAPTIONS, VISUAL_ENCODER_MODEL, VOCAB_PATH,
)
from src.shared.utils import log_reproducibility, setup_reproducibility
from src.shared.vocabulary import Vocabulary


def main() -> None:
    setup_reproducibility(42)
    accelerator = Accelerator()
    log_reproducibility(accelerator, 42)
    vocab = Vocabulary.load(VOCAB_PATH)
    test_df = pd.read_parquet(TEST_DF_PATH)
    dataset = RAGFeatureDataset(
        test_df, TEST_VISUAL_FEATURES_PATH, TEST_I2T_PATH,
        TOP_K_CAPTIONS, TEXT_RAG_CAPTION_SORT_BY,
    )
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=True, collate_fn=collate_fact_batch,
    )
    if not BEST_CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {BEST_CHECKPOINT_PATH}")
    with accelerator.main_process_first():
        fact_encoder = CachedCLIPFactEncoder(VISUAL_ENCODER_MODEL, accelerator.device, FACT_TEXT_BATCH_SIZE)
    model = FactMemoryRAGCaptioner(
        vocab_size=len(vocab), d_model=DMODEL, nheads=NHEADS, nlayers=NLAYERS,
        dropout=DROPOUT, max_length=MAX_LENGTH, pad_idx=vocab.pad_idx(),
        visual_feature_dim=dataset.feature_shape[-1], fact_feature_dim=fact_encoder.output_dim,
        fusion_layers=FACT_FUSION_NLAYERS,
    )
    epoch, _, _ = load_checkpoint(BEST_CHECKPOINT_PATH, model)
    accelerator.print(
        f"Fact Memory RAG; run mode={RUN_MODE}; epoch={epoch}; CLIP Text={VISUAL_ENCODER_MODEL}; "
        f"sort={TEXT_RAG_CAPTION_SORT_BY}; top K captions={TOP_K_CAPTIONS}; "
        f"fusion layers={FACT_FUSION_NLAYERS}; cache=lazy CPU RAM per process"
    )
    model, loader = accelerator.prepare(model, loader)
    captions = generate_captions(
        model, loader, vocab, BEAM_SIZE, MAX_LENGTH, accelerator, fact_encoder, show_progress=True,
        include_cls_token=False,
    )
    if accelerator.is_main_process:
        predictions = [{"imgid": int(imgid), "caption": " ".join(captions[int(imgid)])}
                       for imgid in test_df["imgid"]]
        PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        PREDICTIONS_PATH.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
        accelerator.print(f"Saved {len(predictions):,} captions to {PREDICTIONS_PATH}")
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
