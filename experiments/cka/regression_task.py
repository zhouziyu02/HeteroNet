"""Activity interpolation/extrapolation training and sequence-level CKA.

Run from any directory, e.g.:
    python -m experiments.cka.regression_task --task interpolation --device cuda

Dataset split, masks, normalization, optimizer, schedule, and model settings match
the activity scripts. Data preparation stays on CPU; only batches move to GPU.
The best checkpoint is selected by validation MSE and evaluated on test once.
"""
from __future__ import annotations

import argparse
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from experiments.cka.common import ITSPMRepresentationCapture, save_cka_result
from lib.evaluation import compute_all_losses, evaluation
from lib.parse_datasets import task_mask
from lib.person_activity import PersonActivity, Activity_time_chunk
from lib.physionet import get_data_min_max, variable_time_collate_series
from models.ITSPM import ITSPM


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["interpolation", "extrapolation"], required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda", help="CUDA is required unless CPU is explicitly selected for smoke tests.")
    parser.add_argument("--max-samples", type=int, default=2048, help="Seeded random test windows for CKA; 0 means all.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Evaluate an already trained checkpoint without training.")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted training from output-dir/latest.pt.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "activity")
    parser.add_argument("--max-train-batches", type=int, default=0, help="Smoke-test limit; 0 means full training epochs.")
    parser.add_argument("--max-eval-batches", type=int, default=0, help="Smoke-test limit; 0 means the full validation/test sets.")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        parser.error("epochs, batch-size, and patience must be positive")
    if min(args.max_samples, args.max_train_batches, args.max_eval_batches) < 0:
        parser.error("sample/batch limits must be non-negative")
    if args.resume and args.checkpoint:
        parser.error("--resume and --checkpoint are mutually exclusive")
    args.output_dir = (args.output_dir or ROOT / "experiments" / "cka" / "outputs" / f"{args.task}_activity_seed{args.seed}").resolve()
    args.data_dir = args.data_dir.resolve()
    if args.checkpoint:
        args.checkpoint = args.checkpoint.resolve()
    return args


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def config_for(args: argparse.Namespace) -> argparse.Namespace:
    """Settings from Interpolation/Extrapolation/scripts/activity.sh."""
    return argparse.Namespace(
        task="imputation" if args.task == "interpolation" else "forecasting",
        dataset="activity", model="ITSPM", n=int(1e8), seed=args.seed,
        history=3000, pred_window=1000, mask_rate=0.3, collate="indseq",
        batch_size=args.batch_size, d_model=128, dropout=0.05, lr=5e-4,
        weight_decay=1e-5, n_ref_points=64, n_scales=4, n_mixer_layers=3,
        max_event_tokens=None, max_gap_tokens=None, kernel_type="gaussian",
        use_periodic_branch=1,
    )


def ensure_activity_cache(data_dir: Path) -> None:
    """Run the repository's original preprocessing once, on CPU.

    PersonActivity.download chooses CUDA itself. Isolating preprocessing avoids
    hundreds of thousands of tiny GPU operations and GPU-serialized datasets.
    Existing raw data is reused by the original download_url implementation.
    """
    if (data_dir / "data.pt").exists() or (data_dir / "processed" / "data.pt").exists():
        return
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Preparing Activity from the repository loader: {data_dir}", flush=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    code = (
        "import sys,torch; from lib.person_activity import PersonActivity; "
        "PersonActivity(sys.argv[1],download=True,device=torch.device('cpu'))"
    )
    subprocess.run([sys.executable, "-c", code, str(data_dir)], cwd=ROOT, env=env, check=True)


def collate_activity(batch: list, *, config: argparse.Namespace,
                     data_min: torch.Tensor, data_max: torch.Tensor,
                     time_max: torch.Tensor) -> dict[str, Any]:
    result = variable_time_collate_series(
        batch, config, torch.device("cpu"),
        data_min=data_min, data_max=data_max, time_max=time_max,
    )
    result["record_ids"] = [str(row[0]) for row in batch]
    return result


def load_activity(args: argparse.Namespace, config: argparse.Namespace) -> tuple[dict[str, DataLoader], dict[str, Any]]:
    ensure_activity_cache(args.data_dir)
    raw = PersonActivity(str(args.data_dir), n_samples=config.n, download=False, device=torch.device("cpu"))
    # These are precisely the original parse_datasets.py splits: original
    # records are separated before overlapping windows are made.
    seen, test = train_test_split(raw, train_size=0.8, random_state=42, shuffle=True)
    train, val = train_test_split(seen, train_size=0.75, random_state=42, shuffle=False)
    data_min, data_max, _ = get_data_min_max(seen, torch.device("cpu"))
    time_max = torch.tensor(config.history + config.pred_window)
    collate = partial(collate_activity, config=config, data_min=data_min,
                      data_max=data_max, time_max=time_max)
    raw_splits = {"train": train, "validation": val, "test": test}
    window_splits = {
        name: task_mask(config, Activity_time_chunk(records, config, torch.device("cpu")))
        for name, records in raw_splits.items()
    }
    config.input_dim = int(train[0][2].shape[-1])
    config.enc_in = config.num_types = config.input_dim
    loaders = {
        name: DataLoader(rows, batch_size=args.batch_size, shuffle=(name == "train"),
                         collate_fn=collate, num_workers=0)
        for name, rows in window_splits.items()
    }
    if any(len(loader) == 0 for loader in loaders.values()):
        raise ValueError("Activity has an empty train/validation/test split")
    metadata = {
        "split_seed": 42,
        "record_counts": {k: len(v) for k, v in raw_splits.items()},
        "window_counts": {k: len(v) for k, v in window_splits.items()},
        "original_record_ids": {k: [str(row[0]) for row in v] for k, v in raw_splits.items()},
        "normalization": "repository min/max over train+validation original records (seen_data)",
        "normalization_min": data_min.tolist(), "normalization_max": data_max.tolist(),
        "time_normalization_denominator": int(time_max),
        "history_ms": 3000, "window_ms": 4000, "window_stride_ms": 1000,
        "interpolation_mask_rate": 0.3,
        "mask_rng": "numpy.default_rng(window_index), independently within each split",
        "sample_unit": "original-record 4-second time window; overlapping windows are not independent recordings",
        "batch_collation": "indseq: pack each variable's observed events separately; zero-pad with mask=0",
    }
    return loaders, metadata


def device_batches(loader: DataLoader, device: torch.device, limit: int = 0):
    for i, batch in enumerate(loader):
        if limit and i >= limit:
            break
        yield {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def evaluate(model: ITSPM, loader: DataLoader, device: torch.device, limit: int = 0) -> dict:
    model.eval()
    with torch.no_grad():
        return evaluation(model, iter(device_batches(loader, device, limit)),
                          min(limit, len(loader)) if limit else len(loader))


def cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def save_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_checkpoint(path: Path, config: argparse.Namespace, task: str) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" not in checkpoint:
        raise ValueError("Expected a CKA adapter checkpoint containing state_dict and config")
    if checkpoint.get("task") != task or checkpoint.get("dataset") != "activity":
        raise ValueError("Checkpoint task/dataset do not match this experiment")
    saved = checkpoint.get("config", {})
    for key in ("input_dim", "d_model", "dropout", "n_ref_points", "n_scales", "n_mixer_layers", "history", "mask_rate"):
        if saved.get(key) != getattr(config, key):
            raise ValueError(f"Checkpoint config mismatch for {key}: {saved.get(key)} != {getattr(config, key)}")
    return checkpoint


def train(model: ITSPM, loaders: dict[str, DataLoader], config: argparse.Namespace,
          args: argparse.Namespace, device: torch.device) -> tuple[Path, dict]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    best_path, latest_path = args.output_dir / "best.pt", args.output_dir / "latest.pt"
    best_mse, best_epoch, start_epoch = float("inf"), -1, 0
    elapsed_before = 0.0
    if args.resume:
        state = read_checkpoint(latest_path, config, args.task)
        if state["config"] != vars(config):
            raise ValueError("Resume configuration differs from the original run (including seed and batch size)")
        model.load_state_dict(state["state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        best_mse, best_epoch = state["best_validation_mse"], state["best_epoch"]
        start_epoch = state["epoch"] + 1
        elapsed_before = state.get("training_seconds", 0.0)
        torch.set_rng_state(state["torch_rng_state"])
        if device.type == "cuda" and "cuda_rng_state" in state:
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        print(f"Resuming after epoch {start_epoch}; best epoch {best_epoch + 1}", flush=True)
    start_time = time.monotonic()
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.monotonic()
        model.train()
        losses = []
        for batch_idx, batch in enumerate(device_batches(loaders["train"], device, args.max_train_batches)):
            optimizer.zero_grad(set_to_none=True)
            result = compute_all_losses(model, batch, "activity")
            loss = result["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch={epoch + 1}, batch={batch_idx}")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            if (batch_idx + 1) % 25 == 0:
                print(f"{args.task} epoch {epoch + 1}: batch {batch_idx + 1}/{len(loaders['train'])}, mse={losses[-1]:.6f}", flush=True)
        validation = evaluate(model, loaders["validation"], device, args.max_eval_batches)
        val_mse = float(validation["mse"])
        if not math.isfinite(val_mse):
            raise RuntimeError("Validation MSE is not finite")
        improved = val_mse < best_mse
        if improved:
            best_mse, best_epoch = val_mse, epoch
        base = {
            "state_dict": cpu_state(model), "task": args.task, "dataset": "activity",
            "config": vars(config), "epoch": epoch, "best_epoch": best_epoch,
            "best_validation_mse": best_mse, "validation_metrics": validation,
            "training_seconds": elapsed_before + time.monotonic() - start_time,
            "smoke_test": bool(args.max_train_batches or args.max_eval_batches or device.type != "cuda"),
        }
        if improved:
            save_checkpoint(best_path, base)
        scheduler.step()
        save_checkpoint(latest_path, {
            **base, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            **({"cuda_rng_state": torch.cuda.get_rng_state_all()} if device.type == "cuda" else {}),
        })
        log = {"epoch": epoch + 1, "train_mse": float(np.mean(losses)),
               "validation_mse": val_mse, "best_epoch": best_epoch + 1,
               "seconds": time.monotonic() - epoch_start}
        with (args.output_dir / "training.jsonl").open("a") as handle:
            handle.write(json.dumps(log) + "\n")
        print(json.dumps(log), flush=True)
        if epoch - best_epoch >= args.patience:
            print(f"Early stopping after {args.patience} epochs without validation improvement", flush=True)
            break
    if not best_path.exists():
        raise RuntimeError("No best checkpoint was produced")
    return best_path, {"training_seconds": elapsed_before + time.monotonic() - start_time}


def capture_test(model: ITSPM, loader: DataLoader, args: argparse.Namespace,
                 device: torch.device, metadata: dict) -> dict:
    count = len(loader.dataset)
    selected = np.arange(count)
    if args.max_samples and args.max_samples < count:
        selected = np.sort(np.random.default_rng(args.seed).choice(count, args.max_samples, replace=False))
    capture_loader = DataLoader(Subset(loader.dataset, selected.tolist()), batch_size=args.batch_size,
                                shuffle=False, collate_fn=loader.collate_fn, num_workers=0)
    first, last, valid, ids = [], [], [], []
    model.eval()
    with torch.no_grad(), ITSPMRepresentationCapture(model) as capture:
        for batch in device_batches(capture_loader, device):
            capture.begin_batch(batch["observed_mask"])
            model.forecasting(batch["tp_to_predict"], batch["observed_data"],
                              batch["observed_tp"], batch["observed_mask"])
            representations = capture.end_batch(last="shared_global")
            first.append(representations["first"])
            last.append(representations["last"])
            valid.append(representations["valid"])
            ids.extend(batch["record_ids"])
    metadata["cka_test_sampling"] = {"population_windows": count, "requested_max": args.max_samples,
                                      "selected_windows": len(ids), "seed": args.seed,
                                      "method": "uniform without replacement, sorted indices"}
    return save_cka_result(args.output_dir, torch.cat(first), torch.cat(last), ids, metadata,
                           valid=torch.cat(valid))


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable. In Colab choose Runtime > Change runtime type > GPU. CPU requires explicit --device cpu for smoke tests.")
    if device.type not in {"cuda", "cpu"}:
        raise ValueError("Use a Colab CUDA GPU; CPU is allowed only for explicit smoke tests")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    config = config_for(args)
    loaders, data_metadata = load_activity(args, config)
    model = ITSPM(config).to(device)
    training_metadata = {}
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path, training_metadata = train(model, loaders, config, args, device)
    checkpoint = read_checkpoint(checkpoint_path, config, args.task)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    test_metrics = evaluate(model, loaders["test"], device, args.max_eval_batches)
    metadata = {
        "task": args.task, "dataset": "activity", "seed": args.seed,
        "training_seed": checkpoint["config"]["seed"],
        "device": str(device), "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": str(torch.__version__), "config": vars(config),
        "optimizer": {"name": "AdamW", "lr": config.lr, "weight_decay": config.weight_decay},
        "lr_schedule": {"name": "StepLR", "step_size_epochs": 30, "gamma": 0.5},
        "effective_max_event_tokens": model.max_event_tokens,
        "effective_max_gap_tokens": model.max_gap_tokens,
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": file_sha256(checkpoint_path),
        "best_epoch": checkpoint["best_epoch"] + 1,
        "best_validation_mse": checkpoint["best_validation_mse"], "test_metrics": test_metrics,
        "data": data_metadata, **training_metadata,
        "checkpoint_selection": "lowest validation MSE; test set never used to choose checkpoint",
        "first_endpoint": "encoder.norm output, masked mean over observed time-variable events",
        "last_endpoint": "_encode global_repr = interaction global + kernel_branch global, before QueryTokenReadout",
        "forward_path": "native ITSPM.forecasting; classification cls_fusion is not called",
        "comparison_scope": "first and final shared backbone stages, one row per held-out Activity window",
        "smoke_test": bool(device.type != "cuda" or args.max_train_batches or args.max_eval_batches or checkpoint.get("smoke_test")),
        "training_limits": {"maximum_epochs": args.epochs, "patience": args.patience,
                            "max_train_batches": args.max_train_batches, "max_eval_batches": args.max_eval_batches},
    }
    result = capture_test(model, loaders["test"], args, device, metadata)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    main()
