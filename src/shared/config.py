"""Shared configuration for baseline, image, text and multimodal RAG runs."""

import os
from pathlib import Path

from dotenv import load_dotenv


def safe_mkdir(path: Path) -> None:
    """Create an output directory when writable, including on Kaggle."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass


# ==========================================
# 1. PROJECT & ENVIRONMENT
# ==========================================
ROOT_PATH = Path(__file__).resolve().parent.parent.parent
ROOT_DIR = ROOT_PATH
ENV_PATH = ROOT_PATH / ".env"

if ENV_PATH.exists():
    # Values in .env take precedence over the current process environment.
    load_dotenv(ENV_PATH, override=True)

# Experiment identifier used for both results and checkpoints.
RUN_MODE = os.getenv("RUN_MODE", "baseline")


# ==========================================
# 2. ENCODER MODELS
# ==========================================
VISUAL_ENCODER_MODEL = os.getenv("VISUAL_ENCODER_MODEL", "openai/clip-vit-base-patch32")
RETRIEVAL_OPENCLIP_MODEL_ID = os.getenv("RETRIEVAL_OPENCLIP_MODEL_ID", "hf-hub:apple/DFN5B-CLIP-ViT-H-14")
RETRIEVAL_IMAGE_BATCH_SIZE = int(os.getenv("RETRIEVAL_IMAGE_BATCH_SIZE", "32"))
BLIP_ITM_MODEL_ID = os.getenv("BLIP_ITM_MODEL_ID", "Salesforce/blip-itm-base-coco")
VISUAL_SAFE_NAME = VISUAL_ENCODER_MODEL.replace("/", "-")
RETRIEVAL_SAFE_NAME = RETRIEVAL_OPENCLIP_MODEL_ID.removeprefix("hf-hub:").replace("/", "-")


# ==========================================
# 3. DATASET, SPLITS & VOCABULARY
# ==========================================
DATASET_COCO_PATH = Path(
    os.getenv(
        "DATASET_COCO_PATH",
        str(ROOT_PATH / "dataset" / "mscoco" / "dataset_coco.json"),
    )
)
IMAGES_DIR = Path(
    os.getenv("IMAGES_PATH", str(ROOT_PATH / "dataset" / "mscoco" / "images"))
)

ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", str(ROOT_PATH / "artifacts")))
safe_mkdir(ARTIFACTS_DIR)

SPLITS_DIR = ARTIFACTS_DIR / "splits"
safe_mkdir(SPLITS_DIR)
TRAIN_DF_PATH = SPLITS_DIR / "train_df.parquet"
VAL_DF_PATH = SPLITS_DIR / "val_df.parquet"
TEST_DF_PATH = SPLITS_DIR / "test_df.parquet"

# Preserve the vocabulary used by existing checkpoints.
VOCAB_PATH = Path(os.getenv("VOCAB_PATH", str(ARTIFACTS_DIR / "vocab.json")))
FREQ_THRESHOLD = int(os.getenv("FREQ_THRESHOLD", "5"))


# ==========================================
# 4. PRECOMPUTED VISUAL FEATURES & EMBEDDINGS
# ==========================================
# Patch/CLS features used by the captioning models.
VISUAL_FEATURES_DIR = Path(
    os.getenv(
        "VISUAL_FEATURES_DIR",
        str(ARTIFACTS_DIR / VISUAL_SAFE_NAME / "visual_features"),
    )
)
safe_mkdir(VISUAL_FEATURES_DIR)
TRAIN_VISUAL_FEATURES_PATH = VISUAL_FEATURES_DIR / "train_visual_features.h5"
VAL_VISUAL_FEATURES_PATH = VISUAL_FEATURES_DIR / "val_visual_features.h5"
TEST_VISUAL_FEATURES_PATH = VISUAL_FEATURES_DIR / "test_visual_features.h5"

# Projected global image embeddings used for retrieval.
IMAGE_EMBEDDINGS_DIR = Path(
    os.getenv(
        "IMAGE_EMBEDDINGS_DIR",
        str(ARTIFACTS_DIR / RETRIEVAL_SAFE_NAME / "image_embeddings"),
    )
)
safe_mkdir(IMAGE_EMBEDDINGS_DIR)
TRAIN_IMAGE_EMBEDDINGS_PATH = IMAGE_EMBEDDINGS_DIR / "train_image_embeddings.h5"
VAL_IMAGE_EMBEDDINGS_PATH = IMAGE_EMBEDDINGS_DIR / "val_image_embeddings.h5"
TEST_IMAGE_EMBEDDINGS_PATH = IMAGE_EMBEDDINGS_DIR / "test_image_embeddings.h5"


# ==========================================
# 5. KNOWLEDGE BASES & RETRIEVAL RESULTS
# ==========================================
# Number of captions retrieved and scored with BLIP ITM per query image.
I2T_RETRIEVAL_TOP_K = int(os.getenv("I2T_RETRIEVAL_TOP_K", "10"))
# Number of query images scored together with BLIP ITM per GPU.
BLIP_ITM_IMAGE_BATCH_SIZE = int(os.getenv("BLIP_ITM_IMAGE_BATCH_SIZE", "2"))

# FACTUAL inference batch size per GPU.
SCENE_GRAPH_BATCH_SIZE = int(os.getenv("SCENE_GRAPH_BATCH_SIZE", "64"))

# Number of captions encoded per batch when building the I2T FAISS index.
KB_I2T_INDEX_BATCH_SIZE = int(os.getenv("KB_I2T_INDEX_BATCH_SIZE", "64"))

KB_DIR = ARTIFACTS_DIR / RETRIEVAL_SAFE_NAME / "kb"
safe_mkdir(KB_DIR)
I2T_KB_FAISS_INDEX_PATH = KB_DIR / "kb_i2t_index.faiss"
I2T_KB_METADATA_PATH = KB_DIR / "kb_i2t_metadata.parquet"
I2T_KB_MANIFEST_PATH = KB_DIR / "kb_i2t_manifest.json"
I2I_KB_FAISS_INDEX_PATH = KB_DIR / "kb_i2i_index.faiss"
I2I_KB_METADATA_PATH = KB_DIR / "kb_i2i_metadata.parquet"

RAG_CONTEXTS_DIR = ARTIFACTS_DIR / RETRIEVAL_SAFE_NAME / "rag_contexts"
safe_mkdir(RAG_CONTEXTS_DIR)

# I2T (Image-to-Text): captions/tokens, semantic units/phrases and retrieval metadata.
TRAIN_I2T_PATH = RAG_CONTEXTS_DIR / "train_i2t.parquet"
VAL_I2T_PATH = RAG_CONTEXTS_DIR / "val_i2t.parquet"
TEST_I2T_PATH = RAG_CONTEXTS_DIR / "test_i2t.parquet"

# I2I (Image-to-Image): retrieved image IDs and similarity scores.
TRAIN_I2I_PATH = RAG_CONTEXTS_DIR / "train_i2i.parquet"
VAL_I2I_PATH = RAG_CONTEXTS_DIR / "val_i2i.parquet"
TEST_I2I_PATH = RAG_CONTEXTS_DIR / "test_i2i.parquet"


# ==========================================
# 6. EXPERIMENT OUTPUTS & CHECKPOINTS
# ==========================================
RESULT_DIR = ARTIFACTS_DIR / "results" / RUN_MODE
safe_mkdir(RESULT_DIR)
PREDICTIONS_PATH = RESULT_DIR / "predictions.json"
METRICS_PATH = RESULT_DIR / "metrics.json"

CHECKPOINTS_DIR = Path(os.getenv("CHECKPOINTS_DIR", str(ROOT_PATH / "checkpoints")))
RUN_CHECKPOINT_DIR = CHECKPOINTS_DIR / RUN_MODE
safe_mkdir(RUN_CHECKPOINT_DIR)
LAST_CHECKPOINT_PATH = RUN_CHECKPOINT_DIR / "last_checkpoint.pth"
BEST_CHECKPOINT_PATH = RUN_CHECKPOINT_DIR / "best_checkpoint.pth"


# ==========================================
# 7. MODEL, TRAINING & CAPTION GENERATION
# ==========================================
# Shared captioning architecture.
DMODEL = int(os.getenv("DMODEL", "512"))
NHEADS = int(os.getenv("NHEADS", "8"))
NLAYERS = int(os.getenv("NLAYERS", "4"))
DROPOUT = float(os.getenv("DROPOUT", "0.1"))

# Optimization and data loading. BATCH_SIZE is per GPU;
# global training batch size = BATCH_SIZE * number of GPUs (without accumulation).
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "10"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-4"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "1e-4"))
MAX_GRAD_NORM = float(os.getenv("MAX_GRAD_NORM", "1.0"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))

# Linear warmup followed by cosine decay; LEARNING_RATE is the peak LR.
WARMUP_RATIO = float(os.getenv("WARMUP_RATIO", "0.05"))
WARMUP_START_FACTOR = float(os.getenv("WARMUP_START_FACTOR", "0.01"))
MIN_LR_RATIO = float(os.getenv("MIN_LR_RATIO", "0.05"))

# Target caption length and beam search, independent of RAG context length.
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "40"))
BEAM_SIZE = int(os.getenv("BEAM_SIZE", "5"))


# ==========================================
# 8. RAG MEMORY SETTINGS
# ==========================================
# Image RAG and multimodal RAG.
TOP_K_RAG_IMAGES = int(os.getenv("TOP_K_RAG_IMAGES", "4"))

# Text RAG and multimodal RAG.
CTX_NLAYERS = int(os.getenv("CTX_NLAYERS", "2"))
TOP_K_CAPTIONS = int(os.getenv("TOP_K_CAPTIONS", "4"))
# Text RAG: rank by descending ITM by default; 'clip' preserves retrieval order.
TEXT_RAG_CAPTION_SORT_BY = os.getenv("TEXT_RAG_CAPTION_SORT_BY", "itm").strip().lower()
CTX_TOKENS_PER_CAPTION = int(os.getenv("CTX_TOKENS_PER_CAPTION", "22"))
MAX_CTX_LENGTH = int(
    os.getenv(
        "MAX_CTX_LENGTH",
        str(TOP_K_CAPTIONS * CTX_TOKENS_PER_CAPTION + TOP_K_CAPTIONS),
    )
)
