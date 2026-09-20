"""Paired, sample-level representation capture and centered linear CKA.

The primary estimator is the ordinary (biased HSIC) linear CKA in Kornblith et
al. (ICML 2019), not an average of independently centered minibatch CKAs.
Reference: https://proceedings.mlr.press/v97/kornblith19a.html
Reference implementation:
https://github.com/google-research/google-research/tree/master/representation_similarity

``first`` is one masked mean of observed event embeddings per sample. ``last``
is one global backbone representation for that SAME sample. These are named
architecture stages, not literal first/last Transformer blocks. In particular,
one-block models do not provide a meaningful first-versus-last block contrast.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


ESTIMATOR = "centered_linear_cka_biased_hsic_full_sample_float64"
LAYER_DEFINITIONS = {
    "first": (
        "EventEncoder.norm output [B,L,C,D], arithmetic mean over observed "
        "(time,channel) entries for each sample; excludes mask=0 entries."
    ),
    "shared_global": (
        "PatternInteraction global_repr + KernelSummaryBranch global_repr; "
        "the global output of ITSPM._encode, before task-specific readout."
    ),
    "forward": (
        "Actual ITSPM.forward output; shared_global plus cls_fusion unless "
        "disable_cls_fusion is active. Optional task-facing endpoint."
    ),
}


def _numpy(value: Any, dtype: Any = None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _paired_matrices(first: Any, last: Any) -> tuple[np.ndarray, np.ndarray]:
    first, last = _numpy(first, np.float64), _numpy(last, np.float64)
    if first.ndim != 2 or last.ndim != 2:
        raise ValueError("CKA inputs must be [paired samples, features] matrices.")
    if first.shape[0] != last.shape[0]:
        raise ValueError("CKA inputs must have the same number and ordering of samples.")
    if first.shape[1] == 0 or last.shape[1] == 0:
        raise ValueError("CKA inputs must contain at least one feature.")
    if not np.isfinite(first).all() or not np.isfinite(last).all():
        raise ValueError("CKA inputs contain NaN or infinity.")
    return first, last


def _center_and_scale(features: np.ndarray) -> np.ndarray:
    """Center all rows jointly; scalar scaling protects products from overflow."""
    if np.all(features == features[:1]):
        raise ValueError("CKA is undefined for a constant representation.")
    scale = np.max(np.abs(features))
    if scale == 0:
        raise ValueError("CKA is undefined for a constant representation.")
    centered = features / scale
    centered = centered - centered.mean(axis=0, keepdims=True)
    centered_scale = np.max(np.abs(centered))
    if centered_scale == 0:
        raise ValueError("CKA is undefined for a constant representation.")
    return centered / centered_scale


def linear_cka(first: Any, last: Any) -> float:
    """Compute ||Xc.T Yc||_F² / (||Xc.T Xc||_F ||Yc.T Yc||_F).

    Each row must refer to the same held-out sample in both matrices. Features
    may have different dimensions. Accumulation and centering use float64;
    scalar rescaling leaves linear CKA unchanged. Degenerate inputs raise a
    ValueError instead of returning a misleading zero or adding denominator eps.
    """
    first, last = _paired_matrices(first, last)
    if first.shape[0] < 2:
        raise ValueError("CKA needs at least two paired samples.")
    x, y = _center_and_scale(first), _center_and_scale(last)
    numerator = np.square(x.T @ y).sum(dtype=np.float64)
    denominator = np.linalg.norm(x.T @ x, ord="fro") * np.linalg.norm(y.T @ y, ord="fro")
    if denominator == 0 or not np.isfinite(denominator):
        raise ValueError("CKA has a zero or non-finite denominator.")
    # Cauchy-Schwarz bounds the biased estimator; tolerate roundoff at 0 and 1.
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def cka_statistics(first: Any, last: Any) -> dict[str, Any]:
    """Return JSON-safe CKA results, explicitly marking degenerate comparisons."""
    first, last = _paired_matrices(first, last)
    result = {
        "estimator": ESTIMATOR,
        "n_samples": int(first.shape[0]),
        "first_features": int(first.shape[1]),
        "last_features": int(last.shape[1]),
        "sample_axis": "paired held-out sequences/windows; globally centered once",
        "linear_cka": None,
        "status": "undefined",
    }
    try:
        result["linear_cka"] = linear_cka(first, last)
    except ValueError as exc:
        result["reason"] = str(exc)
    else:
        result["status"] = "ok"
    return result


def deterministic_indices(size: int, max_samples: int | None, seed: int) -> np.ndarray:
    """Select without replacement, sorting the result to preserve dataset order."""
    if size < 0:
        raise ValueError("Dataset size must be nonnegative.")
    if max_samples is None:
        return np.arange(size, dtype=np.int64)
    if max_samples < 1:
        raise ValueError("max_samples must be positive, or None for all samples.")
    if max_samples >= size:
        return np.arange(size, dtype=np.int64)
    return np.sort(np.random.default_rng(seed).choice(size, max_samples, replace=False))


def save_cka_result(
    output_dir: str | Path,
    first: Any,
    last: Any,
    sample_ids: Any,
    metadata: dict[str, Any],
    *,
    valid: Any = None,
    save_representations: bool = True,
) -> dict[str, Any]:
    """Save paired rows, identities and a self-describing CKA report.

    If supplied, valid is a boolean vector for the ORIGINAL unfiltered rows.
    Non-finite rows are errors, not silently removed. Adapters should record
    dataset/split, checkpoint hash, training/selection seeds and layer endpoints
    in metadata. Input IDs must be unique and JSON-safe scalar identifiers.
    """
    first, last = _paired_matrices(first, last)
    ids = _numpy(sample_ids)
    if ids.ndim != 1 or len(ids) != len(first):
        raise ValueError("sample_ids must contain one identifier per original row.")
    if len(set(map(str, ids.tolist()))) != len(ids):
        raise ValueError("sample_ids must be unique; repeated sample rows are not independent.")
    keep = np.ones(len(first), dtype=bool) if valid is None else _numpy(valid)
    if keep.shape != (len(first),) or keep.dtype != np.bool_:
        raise ValueError("valid must be a boolean vector over original sample rows.")
    first, last = first[keep], last[keep]
    result = cka_statistics(first, last)
    result.update({
        "n_samples_before_filter": int(len(ids)),
        "n_dropped_all_missing": int((~keep).sum()),
        "sample_ids": ids[keep].tolist(),
        "dropped_sample_ids": ids[~keep].tolist(),
        "metadata": metadata,
        "representation_file": "representations.npz" if save_representations else None,
    })
    serialized = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_representations:
        # Use strings for heterogeneous IDs, so loading never requires pickle.
        if ids.dtype.kind == "O":
            ids = ids.astype(str)
        np.savez_compressed(output_dir / "representations.npz", first=first, last=last, sample_ids=ids[keep])
    (output_dir / "results.json").write_text(serialized + "\n", encoding="utf-8")
    return result


class ITSPMRepresentationCapture:
    """Non-mutating hooks for the shared models/ITSPM.py implementation.

    Usage::

        backbone.eval()
        with ITSPMRepresentationCapture(backbone) as capture, torch.no_grad():
            capture.begin_batch(mask)
            prediction = backbone.forecasting(query_times, values, times, mask)
            reps = capture.end_batch(last="shared_global")

    ``first`` and ``last`` retain all original B rows; ``valid`` marks samples
    with at least one observed event. Returned features are detached CPU float64
    tensors. Hook return values are None, and hooks never alter model mode,
    parameters, predictions, RNG state or gradients. One begin/end pair must
    surround exactly one backbone invocation. Forecasting calls _encode directly
    so its forward hook does not fire; encoder.norm does fire in both paths.
    The separate EasyTPP ITSPM architecture uses its own task-specific capture.
    """

    def __init__(self, backbone: torch.nn.Module):
        self.backbone = backbone
        self._handles: list[Any] = []
        self._mask: torch.Tensor | None = None
        self._values: dict[str, torch.Tensor] = {}
        self._interaction_global: torch.Tensor | None = None

    def __enter__(self) -> "ITSPMRepresentationCapture":
        if self._handles:
            raise RuntimeError("Representation capture is already active.")
        required = ("encoder", "interaction", "kernel_branch")
        if not all(hasattr(self.backbone, name) for name in required):
            raise TypeError("Capture requires the shared models/ITSPM.py backbone.")
        self._handles = [
            self.backbone.encoder.norm.register_forward_hook(self._event_hook),
            self.backbone.interaction.register_forward_hook(self._interaction_hook),
            self.backbone.kernel_branch.register_forward_hook(self._kernel_hook),
            self.backbone.register_forward_hook(self._forward_hook),
        ]
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._mask = None
        self._interaction_global = None
        self._values.clear()

    def begin_batch(self, mask: torch.Tensor) -> None:
        if not self._handles:
            raise RuntimeError("Use the representation capture as a context manager.")
        if self._mask is not None:
            raise RuntimeError("Call end_batch before beginning another batch.")
        if self.backbone.training:
            raise RuntimeError("Call backbone.eval() before held-out representation capture.")
        if mask.ndim != 3 or not torch.isfinite(mask).all():
            raise ValueError("Observed mask must be a finite [B,L,C] tensor.")
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("Observed mask must contain only zero or one.")
        self._mask = mask.detach().to(device="cpu", dtype=torch.bool).clone()
        self._values.clear()

    def _record(self, name: str, tensor: torch.Tensor) -> None:
        if name in self._values:
            raise RuntimeError("Multiple backbone calls captured in one batch; reset between calls.")
        self._values[name] = tensor.detach().to(device="cpu", dtype=torch.float64).clone()

    def _event_hook(self, module, args, output) -> None:
        if self._mask is None:
            return
        if output.ndim != 4 or tuple(output.shape[:3]) != tuple(self._mask.shape):
            raise ValueError("Event embedding dimensions do not match the supplied observed mask.")
        event = output.detach().to(device="cpu", dtype=torch.float64)
        mask = self._mask.unsqueeze(-1)
        numerator = event.masked_fill(~mask, 0).sum(dim=(1, 2))
        denominator = mask.sum(dim=(1, 2)).clamp(min=1)
        self._record("first", numerator / denominator)

    def _interaction_hook(self, module, args, output) -> None:
        if self._mask is not None:
            self._record("interaction_global", output[0])
            self._interaction_global = output[0].detach()

    def _kernel_hook(self, module, args, output) -> None:
        if self._mask is not None:
            self._record("kernel_global", output[0])
            if self._interaction_global is None:
                raise RuntimeError("Kernel branch ran before pattern interaction.")
            # Match the backbone's addition in its native dtype exactly, then
            # convert the result to float64 for subsequent CKA calculations.
            self._record("shared_global", self._interaction_global + output[0].detach())

    def _forward_hook(self, module, args, output) -> None:
        if self._mask is not None:
            if not isinstance(output, torch.Tensor) or output.ndim != 2:
                raise ValueError("Expected actual ITSPM.forward output to be [B,D].")
            self._record("forward", output)

    def end_batch(self, last: str = "shared_global") -> dict[str, torch.Tensor]:
        if self._mask is None:
            raise RuntimeError("Call begin_batch before end_batch.")
        if last not in ("shared_global", "forward"):
            raise ValueError("last must be 'shared_global' or 'forward'.")
        required = {"first", "shared_global"}
        if last == "forward":
            required.add("forward")
        if not required.issubset(self._values):
            missing = sorted(required.difference(self._values))
            raise RuntimeError(f"Missing captured endpoints {missing}; execute the intended model path first.")
        shared = self._values["shared_global"]
        result = {
            "first": self._values["first"],
            "last": shared if last == "shared_global" else self._values["forward"],
            "valid": self._mask.flatten(start_dim=1).any(dim=1),
            "shared_global": shared,
        }
        if "forward" in self._values:
            result["forward"] = self._values["forward"]
        self._mask = None
        self._interaction_global = None
        self._values = {}
        return result
