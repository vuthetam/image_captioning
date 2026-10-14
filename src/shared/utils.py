import os
import random

import numpy as np
import torch
from accelerate.utils import set_seed
from torch import nn


def setup_reproducibility(seed: int = 42) -> None:
    """Seed RNGs and enable deterministic algorithms before model setup.

    Call before constructing Accelerator or running any CUDA operations.
    """
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    set_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    """Seed NumPy and Python RNGs using the DataLoader worker's Torch seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_seeded_generator(seed: int = 42) -> torch.Generator:
    """Create an independent seeded generator for a DataLoader."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def log_reproducibility(accelerator, seed: int = 42) -> None:
    """Log the active reproducibility and execution settings on the main process."""
    accelerator.print(
        f"seed={seed}; deterministic={torch.are_deterministic_algorithms_enabled()}; "
        f"cudnn benchmark={torch.backends.cudnn.benchmark}; "
        f"cudnn deterministic={torch.backends.cudnn.deterministic}; "
        f"cuBLAS workspace={os.environ['CUBLAS_WORKSPACE_CONFIG']}; "
        f"torch={torch.__version__}; CUDA={torch.version.cuda}; "
        f"processes={accelerator.num_processes}; precision={accelerator.mixed_precision}"
    )


def trainable_parameters(*modules: nn.Module):
    """Yield all parameters that require gradients across the given modules."""
    seen = set()
    for module in modules:
        for parameter in module.parameters():
            if parameter.requires_grad and parameter.data_ptr() not in seen:
                seen.add(parameter.data_ptr())
                yield parameter


def extract_global_embedding(output: object) -> torch.Tensor:
    """Extract projected CLIP embeddings from old and new Transformers outputs."""
    if isinstance(output, torch.Tensor):
        features = output
    else:
        features = None
        for attribute in ("image_embeds", "text_embeds", "pooler_output"):
            value = getattr(output, attribute, None)
            if isinstance(value, torch.Tensor):
                features = value
                break

        # Tuple output can contain last_hidden_state before the pooled embedding.
        if features is None and isinstance(output, (tuple, list)):
            features = next(
                (value for value in output if isinstance(value, torch.Tensor) and value.ndim == 2),
                None,
            )

        if features is None:
            raise TypeError(f"Unsupported CLIP output type: {type(output).__name__}")

    if features.ndim != 2:
        raise ValueError(
            f"CLIP features phải có 2 chiều (batch, dim), nhận được {tuple(features.shape)}"
        )
    return features
