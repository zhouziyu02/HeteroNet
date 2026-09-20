"""Train/reload taxi ITSPM and compare its first/last representation stages.

This adapter intentionally uses the TPP-specific backbone in EasyTPP.  A CKA
row is a valid next-event prediction context, not an individual sparse token.
Only validation log-likelihood is used for checkpoint selection.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
TPP_ROOT = REPO_ROOT / "Temporal Point Process" / "EasyTemporalPointProcess"
for _path in (REPO_ROOT, TPP_ROOT, TPP_ROOT / "examples"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from easy_tpp.config_factory import DataConfig, ModelConfig
from easy_tpp.model.torch_model.torch_itspm import ITSPM
from easy_tpp.preprocess.data_loader import TPPDataLoader
from easy_tpp.utils import set_seed
from taxi_config import DATASET_DEFAULTS, FINAL_GRID_BY_DATASET, make_config


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def data_hashes(cfg):
    return {split: file_sha256(cfg["data"]["taxi"][key])
            for split, key in (("train", "train_dir"), ("dev", "valid_dir"), ("test", "test_dir"))}


def atomic_torch_save(contents, destination):
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
    try:
        torch.save(contents, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def rng_state():
    np_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [np_state[0], np_state[1].tolist(), np_state[2], np_state[3], np_state[4]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    np_state = state["numpy"]
    np.random.set_state((np_state[0], np.asarray(np_state[1], dtype=np.uint32), *np_state[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available():
            raise RuntimeError("This run's RNG state requires CUDA; resume on a Colab GPU runtime.")
        torch.cuda.set_rng_state_all([state.cpu() for state in state["cuda"]])


def resume_signature(cfg):
    trainer = cfg["ITSPM_train"]["trainer_config"]
    return {
        "model_config": cfg["ITSPM_train"]["model_config"],
        "training": {key: trainer[key] for key in ("seed", "batch_size", "optimizer", "learning_rate")},
        "data_sha256": data_hashes(cfg),
    }


def build_config(args):
    """Reproduce taxi.sh hyperparameters while resolving portable data paths."""
    params = copy.deepcopy(DATASET_DEFAULTS["taxi"])
    params.update(FINAL_GRID_BY_DATASET["taxi"][0])
    params["gpu"] = -1 if args.device == "cpu" else int(args.device.partition(":")[2] or 0)
    if args.epochs is not None:
        params["max_epoch"] = args.epochs
    if args.batch_size is not None:
        params["batch_size"] = args.batch_size
    cfg = make_config("taxi", params, args.seed, f"cka_taxi_seed{args.seed}", True)
    cfg["ITSPM_train"]["base_config"]["base_dir"] = str(args.output_dir)
    data_dir = args.data_dir.resolve()
    for key, filename in (("train_dir", "train.pkl"), ("valid_dir", "dev.pkl"), ("test_dir", "test.pkl")):
        path = data_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Taxi split not found: {path}. Set --data-dir to the directory containing all three splits.")
        cfg["data"]["taxi"][key] = str(path)
    # Retain only the selected task in the reproducibility artifact.
    cfg["data"] = {"taxi": cfg["data"]["taxi"]}
    return cfg


def create_model_and_loaders(cfg, device):
    data_cfg = DataConfig.parse_from_yaml_config(cfg["data"]["taxi"])
    experiment = cfg["ITSPM_train"]
    model_cfg = ModelConfig.parse_from_yaml_config(experiment["model_config"])
    model_cfg.gpu = experiment["trainer_config"]["gpu"]
    model_cfg.is_training = True
    model_cfg.num_event_types = data_cfg.data_specs.num_event_types
    model_cfg.num_event_types_pad = data_cfg.data_specs.num_event_types_pad
    model_cfg.pad_token_id = data_cfg.data_specs.pad_token_id
    model_cfg.max_len = data_cfg.data_specs.max_len
    model_cfg.model_id = "ITSPM"
    model = ITSPM(model_cfg).to(device)
    loaders = TPPDataLoader(
        data_config=data_cfg, backend="torch",
        batch_size=experiment["trainer_config"]["batch_size"], shuffle=True,
    )
    return model, loaders


def likelihood_epoch(model, loader, device, optimizer=None):
    """Match EasyTPP's loss/num_event Adam update and event-weighted reporting."""
    model.train(optimizer is not None)
    loss_sum, event_sum = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for encoded in loader:
            batch = list(encoded.to(device).values())
            loss, n_events = model.loglike_loss(batch)
            n_events = int(n_events)
            if n_events <= 0 or not torch.isfinite(loss).item():
                raise RuntimeError("TPP likelihood has zero valid targets or a non-finite value.")
            if optimizer is not None:
                optimizer.zero_grad()
                (loss / n_events).backward()
                optimizer.step()
            loss_sum += float(loss.detach().cpu())
            event_sum += n_events
    if event_sum == 0:
        raise RuntimeError("No valid next-event targets in this split.")
    return {"loglike": -loss_sum / event_sum, "num_events": event_sum}


def train_checkpoint(model, loaders, cfg, args, device):
    trainer_cfg = cfg["ITSPM_train"]["trainer_config"]
    optimizer = torch.optim.Adam(model.parameters(), lr=trainer_cfg["learning_rate"])
    train_loader = loaders.train_loader()
    valid_loader = loaders.valid_loader(shuffle=False)
    checkpoint = args.output_dir / "best_model.pt"
    latest = args.output_dir / "latest.pt"
    best_loglike, best_epoch = -float("inf"), None
    start_epoch = 0
    signature = resume_signature(cfg)
    if args.resume:
        if not latest.is_file():
            raise FileNotFoundError(f"--resume requires {latest}")
        resumed = torch.load(latest, map_location=device, weights_only=True)
        if resumed["resume_signature"] != signature:
            raise ValueError("Resume model, seed, batch size, optimizer, learning rate, or dataset hashes differ from latest.pt.")
        if resumed["best_epoch"] is not None and not checkpoint.is_file():
            raise FileNotFoundError(f"Resume requires the saved validation-best checkpoint: {checkpoint}")
        model.load_state_dict(resumed["model_state_dict"], strict=True)
        optimizer.load_state_dict(resumed["optimizer_state_dict"])
        best_loglike, best_epoch = resumed["best_validation_loglike"], resumed["best_epoch"]
        start_epoch = resumed["epoch"] + 1
        if trainer_cfg["max_epoch"] < start_epoch:
            raise ValueError(f"Requested epochs ({trainer_cfg['max_epoch']}) is below completed epochs ({start_epoch}).")
        restore_rng_state(resumed["rng_state"])
        print(json.dumps({"resume_from_epoch": start_epoch, "best_epoch": best_epoch}), flush=True)

    def save_latest(epoch):
        atomic_torch_save({
            "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch, "best_epoch": best_epoch, "best_validation_loglike": best_loglike,
            "config": cfg, "resume_signature": signature, "rng_state": rng_state(),
        }, latest)

    if not args.resume:
        # An interruption during epoch zero can restart from this initial state.
        save_latest(-1)
    log_path = args.output_dir / "training.jsonl"
    with log_path.open("a" if args.resume else "w", encoding="utf-8") as log:
        for epoch in range(start_epoch, trainer_cfg["max_epoch"]):
            begin = time.perf_counter()
            train = likelihood_epoch(model, train_loader, device, optimizer)
            valid = likelihood_epoch(model, valid_loader, device)
            if valid["loglike"] > best_loglike:
                best_loglike, best_epoch = valid["loglike"], epoch
                atomic_torch_save({
                    "model_state_dict": model.state_dict(), "config": cfg,
                    "seed": args.seed, "epoch": epoch,
                    "selection": "maximum_validation_loglike", "validation": valid,
                    "data_sha256": signature["data_sha256"],
                }, checkpoint)
            row = {"epoch": epoch, "train": train, "validation": valid,
                   "best_epoch": best_epoch, "elapsed_seconds": time.perf_counter() - begin}
            line = json.dumps(row)
            log.write(line + "\n")
            log.flush()
            save_latest(epoch)
            print(line, flush=True)
    if best_epoch is None:
        raise RuntimeError("No validation-selected checkpoint was created; epochs must be positive.")
    return checkpoint


def load_checkpoint(model, path, device, expected_model_config=None):
    # A trusted repository checkpoint contains tensors plus JSON-compatible metadata.
    loaded = torch.load(path, map_location=device, weights_only=True)
    if isinstance(loaded, dict) and "model_state_dict" in loaded:
        saved_config = loaded.get("config", {}).get("ITSPM_train", {})
        saved_architecture = saved_config.get("model_config")
        if (expected_model_config is not None and saved_architecture is not None
                and saved_architecture != expected_model_config):
            raise ValueError("Checkpoint model configuration differs from the selected taxi architecture.")
        model.load_state_dict(loaded["model_state_dict"], strict=True)
        metadata = {key: loaded.get(key) for key in ("seed", "epoch", "selection", "validation")}
        metadata["configured_epochs"] = saved_config.get("trainer_config", {}).get("max_epoch")
        metadata["architecture_verification"] = "saved_model_config" if saved_architecture else "tensor_shapes_only"
        return metadata
    model.load_state_dict(loaded, strict=True)
    return {"selection": "user_provided_checkpoint_unverified_selection", "architecture_verification": "tensor_shapes_only"}


class TPPRepresentationCapture:
    """Capture the actual event encoder output, before sparse-token interaction."""

    def __init__(self, model):
        self.model = model
        self.first = None
        self.valid = None
        self.handle = None

    def __enter__(self):
        self.handle = self.model.event_encoder.register_forward_hook(self._record)
        return self

    def _record(self, module, inputs, output):
        embeddings = output[0].detach().to(dtype=torch.float64)
        mask = inputs[2].detach().to(dtype=torch.float64)
        count = mask.sum(dim=(1, 2))
        self.first = (embeddings * mask.unsqueeze(-1)).sum(dim=(1, 2)) / count.clamp_min(1).unsqueeze(-1)
        self.valid = count > 0

    def __exit__(self, exc_type, exc_value, traceback):
        self.handle.remove()


@torch.no_grad()
def collect_representations(model, loader, device, max_samples, seed):
    """Sample across all test contexts, preserving paired rows and sequence IDs."""
    from experiments.cka.common import deterministic_indices

    total_contexts = sum(max(0, len(seq) - 1) for seq in loader.dataset.type_seqs)
    chosen = np.asarray(deterministic_indices(total_contexts, max_samples, seed), dtype=np.int64)
    if chosen.size < 2:
        raise ValueError("CKA requires at least two held-out prediction contexts.")
    model.eval()
    first_rows, last_rows, valid_rows, sample_ids, sequence_ids = [], [], [], [], []
    context_offset, sequence_offset = 0, 0
    with TPPRepresentationCapture(model) as capture:
        for encoded in loader:
            batch = list(encoded.to(device).values())
            times, _, types, _, _ = batch
            next_valid = types[:, 1:].ne(model.pad_token_id)
            flat_valid = next_valid.flatten().nonzero().flatten()
            context_count = flat_valid.numel()
            left = np.searchsorted(chosen, context_offset)
            right = np.searchsorted(chosen, context_offset + context_count)
            selected_global = chosen[left:right]
            if selected_global.size:
                relative = torch.as_tensor(selected_global - context_offset, device=device)
                flat_indices = flat_valid[relative]
                # This is the same causal context forward pass as loglike_loss.
                last = model(times[:, :-1], types[:, :-1], None).reshape(-1, model.d_model)
                first_rows.append(capture.first[flat_indices].cpu())
                last_rows.append(last[flat_indices].detach().to(dtype=torch.float64, device="cpu"))
                valid_rows.append(capture.valid[flat_indices].cpu())
                padded_length = next_valid.shape[1]
                seq_indices = (flat_indices // padded_length).cpu().tolist()
                event_indices = (flat_indices % padded_length).cpu().tolist()
                for row, prefix_end in zip(seq_indices, event_indices):
                    sample_ids.append(f"taxi:test:sequence={sequence_offset + row}:context_end={prefix_end}")
                    sequence_ids.append(sequence_offset + row)
            context_offset += context_count
            sequence_offset += types.shape[0]
    if context_offset != total_contexts or len(sample_ids) != len(chosen):
        raise RuntimeError("Dataset context indexing disagrees with the padded test loader.")
    return (torch.cat(first_rows), torch.cat(last_rows), torch.cat(valid_rows),
            sample_ids, total_contexts, len(set(sequence_ids)))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=None, help="Default: taxi.sh's 20 epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Default: taxi.sh's batch size 128.")
    parser.add_argument("--device", default="cuda", help="cuda, cuda:N, or cpu (for small smoke checks only).")
    parser.add_argument("--max-samples", type=int, default=2048)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "cka" / "tpp_taxi")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Reuse a trusted matching checkpoint without training.")
    parser.add_argument("--resume", action="store_true", help="Continue interrupted training from output-dir/latest.pt.")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "Temporal Point Process" / "taxi")
    args = parser.parse_args(argv)
    if args.device != "cpu" and not (args.device == "cuda" or args.device.startswith("cuda:")):
        parser.error("--device must be cpu, cuda, or cuda:N")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive; use --checkpoint to skip training")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.max_samples < 2:
        parser.error("--max-samples must be at least 2")
    if args.resume and args.checkpoint:
        parser.error("--resume and --checkpoint are mutually exclusive")
    return args


def main(argv=None):
    from experiments.cka.common import save_cka_result

    args = parse_args(argv)
    device = torch.device("cuda:0" if args.device == "cuda" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Select a Colab GPU runtime before running this experiment.")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume and not args.checkpoint:
        existing = [args.output_dir / name for name in ("latest.pt", "best_model.pt", "training.jsonl")]
        if any(path.exists() for path in existing):
            raise FileExistsError("This output directory contains an existing training run. Use --resume or choose a new --output-dir.")
    set_seed(args.seed)
    cfg = build_config(args)
    (args.output_dir / "effective_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    model, loaders = create_model_and_loaders(cfg, device)
    checkpoint = args.checkpoint.resolve() if args.checkpoint else train_checkpoint(model, loaders, cfg, args, device)
    checkpoint_metadata = load_checkpoint(model, checkpoint, device, cfg["ITSPM_train"]["model_config"])
    # The held-out test split is first evaluated only after reloading this checkpoint.
    test_loader = loaders.test_loader()
    test_metrics = likelihood_epoch(model, test_loader, device)
    first, last, valid, ids, total, sampled_sequences = collect_representations(
        model, test_loader, device, args.max_samples, args.seed,
    )
    metadata = {
        "task": "tpp", "dataset": "taxi", "split": "test", "seed": args.seed,
        "device": str(device), "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint), "data_sha256": data_hashes(cfg),
        "is_smoke_test": (device.type == "cpu"
                          or cfg["ITSPM_train"]["trainer_config"]["max_epoch"] < DATASET_DEFAULTS["taxi"]["max_epoch"]
                          or (args.checkpoint is not None and (checkpoint_metadata.get("configured_epochs") or 20) < 20)),
        "checkpoint_metadata": checkpoint_metadata, "test_metrics": test_metrics,
        "first_layer": "event_encoder.norm output (observed event/channel masked mean)",
        "last_layer": "interaction.global_proj output; actual TPP.forward state before intensity head",
        "sample_unit": "one causal context for a valid next-event prediction",
        "sample_selection": "seeded uniform sampling without replacement across all test contexts",
        "total_test_contexts": total, "sampled_test_sequences": sampled_sequences,
        "total_test_sequences": len(test_loader.dataset),
        "pooling": "first: observed entries in each causal window; last: native global context vector",
        "architecture_note": "TPP uses its own ITSPM implementation with one mixer block and no kernel branch.",
        "training_note": "Original taxi.sh architecture, likelihood, Adam and epoch count; only validation loglike is evaluated during training. Stochastic thinning prediction is omitted, so RNG trajectory differs from the benchmark runner.",
        "dependence_note": "Multiple contexts may come from the same test sequence; CKA is descriptive, with no independence-based confidence interval.",
        "config": cfg,
    }
    result = save_cka_result(args.output_dir, first, last, ids, metadata, valid=valid)
    print(json.dumps({
        "task": "tpp", "dataset": "taxi", "linear_cka": result.get("linear_cka"),
        "status": result.get("status"), "n_samples": result.get("n_samples"),
        "results_file": str(args.output_dir / "results.json"),
    }), flush=True)
    return result


if __name__ == "__main__":
    main()
