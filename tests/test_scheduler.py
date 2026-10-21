import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.scheduler import AcceleratedScheduler
import torch
from torch.optim.lr_scheduler import LambdaLR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.checkpoint import load_checkpoint, save_checkpoint
from src.shared.scheduler import (
    create_accelerated_warmup_cosine_scheduler,
    optimizer_steps,
    unwrap_scheduler,
)


class LibrarySchedulerTests(unittest.TestCase):
    def make_objects(self, total_updates=20, num_processes=1, accelerator=None):
        model = torch.nn.Linear(2, 1).double()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = create_accelerated_warmup_cosine_scheduler(
            optimizer,
            total_updates=total_updates,
            num_processes=num_processes,
            warmup_ratio=0.2,
            warmup_start_factor=0.01,
            min_lr_ratio=0.05,
        )
        if accelerator is not None:
            model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
        return model, optimizer, scheduler

    def update(self, model, optimizer, scheduler, count):
        rates = []
        inputs = torch.tensor([[0.5, 1.0]], dtype=torch.float64)
        for _ in range(count):
            optimizer.zero_grad(set_to_none=True)
            loss = (model(inputs) - 0.2).square().sum()
            loss.backward()
            optimizer.step()
            scheduler.step()
            rates.append(optimizer.param_groups[0]["lr"])
        return rates

    def test_uses_transformers_library_curve_with_configured_endpoints(self):
        model, optimizer, scheduler = self.make_objects(total_updates=100)
        self.assertIsInstance(scheduler, LambdaLR)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-5)

        used_rates = []
        for _ in range(100):
            used_rates.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()

        self.assertAlmostEqual(used_rates[0], 1e-5)
        self.assertAlmostEqual(max(used_rates), 1e-3)
        self.assertAlmostEqual(used_rates[-1], 5e-5)
        self.assertEqual(scheduler.schedule_config["warmup_updates"], 20)

    def test_accelerated_wrapper_advances_inner_scheduler_per_process(self):
        accelerator = Accelerator(
            cpu=True, dataloader_config=DataLoaderConfiguration(split_batches=False)
        )
        model, optimizer, curve = self.make_objects(
            total_updates=20, num_processes=2
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        curve = create_accelerated_warmup_cosine_scheduler(
            optimizer, total_updates=20, num_processes=2,
            warmup_ratio=0.2, warmup_start_factor=0.01, min_lr_ratio=0.05,
        )
        optimizer = accelerator.prepare(optimizer)
        scheduler = AcceleratedScheduler(curve, optimizer, split_batches=False)
        with patch(
            "accelerate.scheduler.AcceleratorState",
            return_value=SimpleNamespace(num_processes=2),
        ):
            optimizer.step()
            scheduler.step()
        self.assertEqual(curve.last_epoch, 2)
        self.assertEqual(optimizer_steps(curve), 1)

    def test_checkpoint_resume_matches_uninterrupted_training(self):
        accelerator = Accelerator(
            cpu=True, dataloader_config=DataLoaderConfiguration(split_batches=False)
        )
        torch.manual_seed(42)
        reference, optimizer, scheduler = self.make_objects(accelerator=accelerator)
        initial_weights = copy.deepcopy(reference.state_dict())
        reference_rates = self.update(reference, optimizer, scheduler, 20)

        with tempfile.TemporaryDirectory() as temp:
            model, optimizer, scheduler = self.make_objects(accelerator=accelerator)
            model.load_state_dict(initial_weights)
            rates = self.update(model, optimizer, scheduler, 7)
            path = Path(temp) / "checkpoint.pth"
            save_checkpoint(
                path, model, optimizer, 1, 1.0, 0.9,
                accelerator=accelerator, scheduler=scheduler,
            )

            restored, restored_optimizer, restored_scheduler = self.make_objects()
            load_checkpoint(path, restored, restored_optimizer, scheduler=restored_scheduler)
            restored, restored_optimizer, restored_scheduler = accelerator.prepare(
                restored, restored_optimizer, restored_scheduler
            )
            rates += self.update(restored, restored_optimizer, restored_scheduler, 13)
            self.assertEqual(rates, reference_rates)
            for key, value in reference.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)

    def test_checkpoint_rejects_changed_schedule(self):
        with tempfile.TemporaryDirectory() as temp:
            model, optimizer, scheduler = self.make_objects()
            path = Path(temp) / "checkpoint.pth"
            save_checkpoint(path, model, optimizer, 1, 1.0, 0.9, scheduler=scheduler)
            other, other_optimizer, other_scheduler = self.make_objects(total_updates=30)
            with self.assertRaisesRegex(ValueError, "settings differ"):
                load_checkpoint(path, other, other_optimizer, scheduler=other_scheduler)

    def test_wrapper_skips_unsynchronized_updates(self):
        accelerator = Accelerator(
            cpu=True, dataloader_config=DataLoaderConfiguration(split_batches=False)
        )
        model, optimizer, curve = self.make_objects(accelerator=accelerator)
        with patch.object(curve.gradient_state, "sync_gradients", False):
            optimizer.step()
            curve.step()
        self.assertEqual(unwrap_scheduler(curve).last_epoch, 0)


if __name__ == "__main__":
    unittest.main()
