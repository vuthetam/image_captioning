from pathlib import Path
import torch

from src.shared.scheduler import optimizer_steps, unwrap_scheduler

def save_checkpoint(
    path: str | Path,
    model,
    optimizer,
    epoch: int,
    train_loss: float,
    best_val_loss: float,
    accelerator=None,
    scheduler=None,
) -> Path:
    """Save a checkpoint from raw or prepared objects.

    Pass ``accelerator`` when ``model`` is prepared; it is optional for a raw model.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if accelerator is not None:
        model_state_dict = accelerator.get_state_dict(model)
    else:
        model_state_dict = model.state_dict()

    checkpoint_state = {
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "best_val_loss": float(best_val_loss),
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
    }

    if scheduler is not None:
        lr_schedule = unwrap_scheduler(scheduler)
        checkpoint_state["scheduler_state_dict"] = scheduler.state_dict()
        checkpoint_state["scheduler_config"] = lr_schedule.schedule_config
        checkpoint_state["global_step"] = optimizer_steps(lr_schedule)

    torch.save(checkpoint_state, path)
    return path


def load_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    strict: bool = True,
    scheduler=None,
):
    """Load a checkpoint before preparing the model and scheduler.

    The model must be raw or unwrapped; the optimizer may be raw or prepared.
    """
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)

    if scheduler is not None:
        if optimizer is None or checkpoint.get("optimizer_state_dict") is None:
            raise ValueError("An optimizer state is required to resume the LR schedule.")
        scheduler_state = checkpoint.get("scheduler_state_dict")
        if scheduler_state is None:
            raise ValueError(
                "This checkpoint has no LR scheduler state. Use a new RUN_MODE "
                "for warmup/cosine training, or load weights without a scheduler."
            )
        saved_config = dict(checkpoint.get("scheduler_config") or {})
        if saved_config != scheduler.schedule_config:
            raise ValueError(
                "LR schedule settings differ from this checkpoint. Keep the peak LR, "
                "scheduler ratios and total update count unchanged when resuming; "
                "use a new RUN_MODE for a new schedule."
            )
        global_step = checkpoint.get("global_step")
        if (
            not isinstance(global_step, int)
            or global_step * scheduler.schedule_config["num_processes"]
            != scheduler_state.get("last_epoch")
        ):
            raise ValueError("Checkpoint global_step and scheduler state do not match.")
        scheduler.load_state_dict(scheduler_state)

    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = int(checkpoint.get("epoch", 0))
    train_loss = float(checkpoint.get("train_loss", 0.0))
    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    return epoch, train_loss, best_val_loss
