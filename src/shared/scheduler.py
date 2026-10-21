"""Helpers for Hugging Face's warmup/cosine scheduler and Accelerate wrapper."""

import math

from accelerate.scheduler import AcceleratedScheduler
from transformers import get_cosine_with_min_lr_schedule_with_warmup_lr_rate


def create_accelerated_warmup_cosine_scheduler(
    optimizer,
    total_updates: int,
    num_processes: int,
    warmup_ratio: float = 0.05,
    warmup_start_factor: float = 0.01,
    min_lr_ratio: float = 0.05,
):
    """Create the library scheduler, with counts matching Accelerate's ticks.

    AcceleratedScheduler advances the inner scheduler once per process when
    split_batches=False. Its training horizon is therefore expressed in those
    inner ticks; warmup is rounded in optimizer updates to stay aligned across
    process counts.
    """
    if not isinstance(total_updates, int) or total_updates < 1:
        raise ValueError("total_updates must be a positive integer.")
    if not isinstance(num_processes, int) or num_processes < 1:
        raise ValueError("num_processes must be a positive integer.")
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be in [0, 1).")
    if not 0.0 < warmup_start_factor <= 1.0:
        raise ValueError("warmup_start_factor must be in (0, 1].")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in [0, 1].")

    warmup_updates = min(math.ceil(total_updates * warmup_ratio), total_updates - 1)
    warmup_scheduler_steps = warmup_updates * num_processes
    total_scheduler_steps = total_updates * num_processes

    scheduler = get_cosine_with_min_lr_schedule_with_warmup_lr_rate(
        optimizer,
        num_warmup_steps=warmup_scheduler_steps,
        num_training_steps=total_scheduler_steps,
        min_lr_rate=min_lr_ratio,
        warmup_lr_rate=warmup_start_factor,
    )
    # LambdaLR does not serialize its callable, so keep its inputs alongside
    # the state for reliable checkpoint compatibility checks.
    scheduler.schedule_config = {
        "name": "transformers_cosine_with_min_lr_schedule_with_warmup_lr_rate",
        "total_updates": total_updates,
        "num_processes": num_processes,
        "total_scheduler_steps": total_scheduler_steps,
        "warmup_ratio": warmup_ratio,
        "warmup_updates": warmup_updates,
        "warmup_scheduler_steps": warmup_scheduler_steps,
        "warmup_start_factor": warmup_start_factor,
        "min_lr_ratio": min_lr_ratio,
        "peak_lrs": list(scheduler.base_lrs),
    }
    return scheduler


def unwrap_scheduler(scheduler):
    """Return the underlying PyTorch scheduler when Accelerate wrapped it."""
    return scheduler.scheduler if isinstance(scheduler, AcceleratedScheduler) else scheduler


def optimizer_steps(scheduler) -> int:
    """Report optimizer updates represented by the scheduler's internal ticks."""
    lr_schedule = unwrap_scheduler(scheduler)
    return lr_schedule.last_epoch // lr_schedule.schedule_config["num_processes"]
