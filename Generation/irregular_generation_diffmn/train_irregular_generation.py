import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from scipy.stats import entropy


TASK_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASK_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.ITSPM import ITSPM  # noqa: E402


DIFF_MN_TABLE1 = {
    "ds": {
        0.3: {"sines": 0.105, "stocks": 0.142, "energy": 0.422, "mujoco": 0.293},
        0.5: {"sines": 0.128, "stocks": 0.137, "energy": 0.487, "mujoco": 0.375},
        0.7: {"sines": 0.182, "stocks": 0.106, "energy": 0.497, "mujoco": 0.393},
    },
    "mdd": {
        0.3: {"sines": 0.953, "stocks": 0.25, "energy": 0.270, "mujoco": 0.347},
        0.5: {"sines": 1.093, "stocks": 0.281, "energy": 0.252, "mujoco": 0.318},
        0.7: {"sines": 1.308, "stocks": 0.299, "energy": 0.279, "mujoco": 0.297},
    },
    "kl": {
        0.3: {"sines": 0.013, "stocks": 0.074, "energy": 0.020, "mujoco": 0.021},
        0.5: {"sines": 0.023, "stocks": 0.094, "energy": 0.022, "mujoco": 0.009},
        0.7: {"sines": 0.033, "stocks": 0.091, "energy": 0.017, "mujoco": 0.014},
    },
}


@dataclass
class ITSPMConfig:
    input_dim: int
    d_model: int = 64
    dropout: float = 0.1
    n_ref_points: int = 24
    n_scales: int = 2
    n_mixer_layers: int = 1
    max_event_tokens: int | None = None
    max_gap_tokens: int | None = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_sine_dataset(n_samples: int, seq_len: int, channels: int, seed: int) -> np.ndarray:
    # Matches Diff-MN/TimeGAN style Sines: frequency and phase in [0, 0.1],
    # normalized to [0, 1].
    rng = np.random.default_rng(seed)
    data = []
    for i in range(n_samples):
        temp = []
        for c in range(channels):
            freq = rng.uniform(0.0, 0.1)
            phase = rng.uniform(0.0, 0.1)
            temp.append([np.sin(freq * j + phase) for j in range(seq_len)])
        data.append(((np.asarray(temp).T + 1.0) * 0.5).astype(np.float32))
    return np.asarray(data, dtype=np.float32)


def make_polynomial_dataset(n_samples: int, seq_len: int, channels: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(-1.0, 1.0, seq_len, dtype=np.float32)
    powers = np.stack([t, t**2, t**3], axis=1)
    data = np.zeros((n_samples, seq_len, channels), dtype=np.float32)
    for i in range(n_samples):
        shared = rng.normal(0.0, 0.45, size=(3,))
        for c in range(channels):
            coeff = shared + rng.normal(0.0, 0.2, size=(3,))
            data[i, :, c] = powers @ coeff + rng.normal(0.0, 0.03, size=seq_len)
    return data


def minmax_scale(data: np.ndarray) -> np.ndarray:
    data_min = data.min(axis=0, keepdims=True)
    data_max = data.max(axis=0, keepdims=True)
    return (data - data_min) / (data_max - data_min + 1e-7)


def load_windowed_csv(path: Path, seq_len: int, n_samples: int | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing real dataset CSV: {path}. Place the corresponding Diff-MN "
            "stock_data.csv or energy_data.csv under --data_root. "
            "No synthetic substitute is generated for Stocks or Energy."
        )
    try:
        raw = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2).astype(np.float32)
    except ValueError as exc:
        raise ValueError(f"{path}: expected numeric CSV columns after one header row.") from exc
    if raw.shape[0] < seq_len or raw.shape[1] == 0:
        raise ValueError(f"{path}: need at least {seq_len} rows and one numeric channel; got {raw.shape}.")
    if not np.isfinite(raw).all():
        raise ValueError(f"{path}: dataset contains NaN or infinity.")
    raw = minmax_scale(raw[::-1])
    windows = np.stack([raw[i:i + seq_len] for i in range(0, len(raw) - seq_len + 1)], axis=0)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(windows))
    windows = windows[idx]
    if n_samples is not None and n_samples > 0:
        windows = windows[:n_samples]
        idx = idx[:n_samples]
    return windows.astype(np.float32), idx


def make_sample_timestep_mask(shape: tuple[int, int, int], missing: float, seed: int = 56789) -> np.ndarray:
    # Official Diff-MN loader drops full timesteps per sample for Sines/MuJoCo.
    n, seq_len, channels = shape
    mask = np.ones(shape, dtype=np.float32)
    n_drop = int(seq_len * missing)
    generator = torch.Generator().manual_seed(seed)
    for i in range(n):
        if n_drop > 0:
            removed = torch.randperm(seq_len, generator=generator)[:n_drop].numpy()
            mask[i, removed, :] = 0.0
    mask[:, 0, :] = 1.0
    mask[:, -1, :] = 1.0
    return mask


def make_global_window_mask(total_len: int, channels: int, seq_len: int, missing: float, idx: np.ndarray) -> np.ndarray:
    # Official Diff-MN loader drops full rows on the raw continuous time axis
    # for stock/energy, then applies the same shuffled window order.
    full_mask = np.ones((total_len, channels), dtype=np.float32)
    n_drop = int(total_len * missing)
    if n_drop > 0:
        generator = torch.Generator().manual_seed(56789)
        removed = torch.randperm(total_len, generator=generator)[:n_drop].numpy()
        full_mask[removed, :] = 0.0
    windows = np.stack([full_mask[i:i + seq_len] for i in range(0, total_len - seq_len + 1)], axis=0)
    return windows[idx].astype(np.float32)


def load_table1_dataset(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, bool]:
    dataset = "sines" if args.dataset == "sine" else args.dataset
    if dataset == "sines":
        data = make_sine_dataset(args.n_samples, args.seq_len, args.channels, args.seed)
        return data, make_sample_timestep_mask(data.shape, args.missing), True
    if dataset == "stocks":
        path = Path(args.data_root) / "stock_data.csv"
        data, idx = load_windowed_csv(path, args.seq_len, args.n_samples, args.seed)
        raw = np.loadtxt(path, delimiter=",", skiprows=1).astype(np.float32)
        mask = make_global_window_mask(len(raw), data.shape[-1], args.seq_len, args.missing, idx)
        return data, mask, True
    if dataset == "energy":
        path = Path(args.data_root) / "energy_data.csv"
        data, idx = load_windowed_csv(path, args.seq_len, args.n_samples, args.seed)
        raw = np.loadtxt(path, delimiter=",", skiprows=1).astype(np.float32)
        mask = make_global_window_mask(len(raw), data.shape[-1], args.seq_len, args.missing, idx)
        return data, mask, True
    if dataset == "mujoco":
        pt = Path(args.data_root) / f"mujoco_training_{args.seq_len}.pt"
        if not pt.is_file():
            raise FileNotFoundError(
                f"Missing real MuJoCo trajectory file: {pt}. "
                "From the repository root, install the physics dependencies with "
                "'python -m experiments.cka.install_physics', then run "
                f"'python -m experiments.cka.prepare_mujoco --lengths {args.seq_len}', "
                "or provide the matching trajectory file under --data_root. "
                "Synthetic proxy fallback is disabled."
            )
        tensor = torch.load(pt, map_location="cpu", weights_only=True)
        if (not isinstance(tensor, torch.Tensor) or tensor.ndim != 3
                or tuple(tensor.shape[1:]) != (args.seq_len, 14) or tensor.shape[0] == 0):
            raise ValueError(f"{pt}: expected a nonempty MuJoCo tensor shaped [N,{args.seq_len},14].")
        data = tensor.detach().cpu().numpy()
        if not np.isfinite(data).all():
            raise ValueError(f"{pt}: trajectory data contains NaN or infinity.")
        if args.n_samples > 0:
            data = data[:args.n_samples]
        data = minmax_scale(data.reshape(-1, data.shape[-1])).reshape(data.shape).astype(np.float32)
        return data, make_sample_timestep_mask(data.shape, args.missing), True
    if dataset == "polynomial":
        data = make_polynomial_dataset(args.n_samples, args.seq_len, args.channels, args.seed)
        return data, make_sample_timestep_mask(data.shape, args.missing), False
    raise ValueError(f"Unknown dataset: {args.dataset}")


def normalize_train_test(data: np.ndarray, train_frac: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split = int(len(data) * train_frac)
    mean = data[:split].mean(axis=(0, 1), keepdims=True)
    std = data[:split].std(axis=(0, 1), keepdims=True) + 1e-6
    return (data - mean) / std, mean, std, np.arange(split)


def denormalize(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (data * std + mean).astype(np.float32)


class SinTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(torch.arange(half) * -(math.log(10000.0) / max(half - 1, 1)))
        self.register_buffer("freqs", freqs)
        self.proj = nn.Linear(1 + 2 * half, dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        angle = t.unsqueeze(-1) * self.freqs
        return self.proj(torch.cat([t.unsqueeze(-1), torch.sin(angle), torch.cos(angle)], dim=-1))


class ITSPMLatentAutoencoder(nn.Module):
    def __init__(self, cfg: ITSPMConfig, seq_len: int, channels: int, latent_dim: int):
        super().__init__()
        self.seq_len = seq_len
        self.channels = channels
        self.backbone = ITSPM(cfg)
        self.to_latent = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, latent_dim),
        )
        self.time_emb = SinTimeEmbedding(latent_dim)
        self.decoder = nn.Sequential(
            nn.LayerNorm(latent_dim * 2),
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, channels),
        )

    def encode(self, times: torch.Tensor, x_obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.backbone(times, x_obs, mask)
        return self.to_latent(h)

    def decode(self, z: torch.Tensor, seq_len: int | None = None) -> torch.Tensor:
        if seq_len is None:
            seq_len = self.seq_len
        bsz = z.size(0)
        t = torch.linspace(0.0, 1.0, seq_len, device=z.device).unsqueeze(0).expand(bsz, seq_len)
        te = self.time_emb(t)
        zt = z.unsqueeze(1).expand(bsz, seq_len, z.size(-1))
        return self.decoder(torch.cat([zt, te], dim=-1))

    def forward(self, times: torch.Tensor, x_obs: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(times, x_obs, mask)
        return self.decode(z), z


class LatentDenoiser(nn.Module):
    def __init__(self, latent_dim: int, hidden: int, n_steps: int):
        super().__init__()
        self.time_emb = nn.Embedding(n_steps, hidden)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_t, self.time_emb(t)], dim=-1))


class SequenceDenoiser(nn.Module):
    def __init__(self, channels: int, hidden: int, n_steps: int):
        super().__init__()
        self.time_emb = nn.Embedding(n_steps, hidden)
        self.in_proj = nn.Conv1d(channels, hidden, 3, padding=1)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(4, hidden),
                nn.SiLU(),
                nn.Conv1d(hidden, hidden, 3, padding=1),
                nn.GroupNorm(4, hidden),
                nn.SiLU(),
                nn.Conv1d(hidden, hidden, 3, padding=1),
            )
            for _ in range(4)
        ])
        self.t_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Conv1d(hidden, channels, 3, padding=1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x_t.transpose(1, 2))
        te = self.t_proj(self.time_emb(t)).unsqueeze(-1)
        for block in self.blocks:
            h = h + block(h + te)
        return self.out_proj(h).transpose(1, 2)


class ITSPMConditionedDenoiser(nn.Module):
    def __init__(self, cfg: ITSPMConfig, channels: int, hidden: int, n_steps: int):
        super().__init__()
        self.itspm = ITSPM(cfg)
        self.time_emb = nn.Embedding(n_steps, hidden)
        self.cond_proj = nn.Linear(cfg.d_model, hidden)
        self.in_proj = nn.Conv1d(channels * 3, hidden, 3, padding=1)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(4, hidden),
                nn.SiLU(),
                nn.Conv1d(hidden, hidden, 3, padding=1),
                nn.GroupNorm(4, hidden),
                nn.SiLU(),
                nn.Conv1d(hidden, hidden, 3, padding=1),
            )
            for _ in range(4)
        ])
        self.out_proj = nn.Conv1d(hidden, channels, 3, padding=1)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        obs: torch.Tensor,
        mask: torch.Tensor,
        obs_times: torch.Tensor,
    ) -> torch.Tensor:
        cond = self.cond_proj(self.itspm(obs_times, obs, mask)).unsqueeze(-1)
        te = self.time_emb(t).unsqueeze(-1)
        denoise_in = torch.cat([x_t, obs, mask], dim=-1)
        h = self.in_proj(denoise_in.transpose(1, 2))
        for block in self.blocks:
            h = h + block(h + cond + te)
        return self.out_proj(h).transpose(1, 2)


class LatentDDPM:
    def __init__(self, n_steps: int, beta_start: float, beta_end: float, device: torch.device):
        self.n_steps = n_steps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, n_steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, z0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        ab = self.alpha_bar[t].view(-1, *([1] * (z0.dim() - 1)))
        return torch.sqrt(ab) * z0 + torch.sqrt(1.0 - ab) * noise

    def predict_x0(self, z_t: torch.Tensor, t: torch.Tensor, pred_noise: torch.Tensor) -> torch.Tensor:
        ab = self.alpha_bar[t].view(-1, *([1] * (z_t.dim() - 1)))
        return (z_t - torch.sqrt(1.0 - ab) * pred_noise) / (torch.sqrt(ab) + 1e-8)

    @torch.no_grad()
    def sample(self, denoiser: LatentDenoiser, n_samples: int, latent_dim: int) -> torch.Tensor:
        z = torch.randn(n_samples, latent_dim, device=self.device)
        for step in reversed(range(self.n_steps)):
            t = torch.full((n_samples,), step, dtype=torch.long, device=self.device)
            eps = denoiser(z, t)
            beta = self.betas[step]
            alpha = self.alphas[step]
            ab = self.alpha_bar[step]
            z = (z - beta / torch.sqrt(1.0 - ab) * eps) / torch.sqrt(alpha)
            if step > 0:
                z = z + torch.sqrt(beta) * torch.randn_like(z)
        return z


class SequenceDDPM(LatentDDPM):
    @torch.no_grad()
    def sample_sequence(self, denoiser: SequenceDenoiser, n_samples: int, seq_len: int, channels: int) -> torch.Tensor:
        x = torch.randn(n_samples, seq_len, channels, device=self.device)
        for step in reversed(range(self.n_steps)):
            t = torch.full((n_samples,), step, dtype=torch.long, device=self.device)
            eps = denoiser(x, t)
            beta = self.betas[step]
            alpha = self.alphas[step]
            ab = self.alpha_bar[step]
            x = (x - beta / torch.sqrt(1.0 - ab) * eps) / torch.sqrt(alpha)
            if step > 0:
                x = x + torch.sqrt(beta) * torch.randn_like(x)
        return x

    @torch.no_grad()
    def sample_conditioned(
        self,
        denoiser: ITSPMConditionedDenoiser,
        obs: torch.Tensor,
        mask: torch.Tensor,
        obs_times: torch.Tensor,
        clamp_observed: bool,
    ) -> torch.Tensor:
        n_samples, seq_len, channels = obs.shape
        x = torch.randn(n_samples, seq_len, channels, device=self.device)
        if clamp_observed:
            x = x * (1.0 - mask) + obs * mask
        for step in reversed(range(self.n_steps)):
            t = torch.full((n_samples,), step, dtype=torch.long, device=self.device)
            eps = denoiser(x, t, obs, mask, obs_times)
            beta = self.betas[step]
            alpha = self.alphas[step]
            ab = self.alpha_bar[step]
            x = (x - beta / torch.sqrt(1.0 - ab) * eps) / torch.sqrt(alpha)
            if step > 0:
                x = x + torch.sqrt(beta) * torch.randn_like(x)
            if clamp_observed:
                x = x * (1.0 - mask) + obs * mask
        return x


def train_direct_diffusion(
    args: argparse.Namespace,
    train_data: np.ndarray,
    test_data: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, dict, nn.Module]:
    ddpm = SequenceDDPM(args.diffusion_steps, args.beta_start, args.beta_end, device)
    denoiser = SequenceDenoiser(args.channels, args.diffusion_hidden, args.diffusion_steps).to(device)
    opt = torch.optim.AdamW(denoiser.parameters(), lr=args.diff_lr, weight_decay=args.weight_decay)
    x_train = torch.from_numpy(train_data.astype(np.float32)).to(device)
    for epoch in range(1, args.diff_epochs + 1):
        losses = []
        denoiser.train()
        for ids_np in iter_batches(len(train_data), args.batch_size, True, args.seed + epoch):
            x0 = x_train[torch.as_tensor(ids_np, device=device)]
            tt = torch.randint(0, args.diffusion_steps, (x0.size(0),), device=device)
            noise = torch.randn_like(x0)
            xt = ddpm.q_sample(x0, tt, noise)
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(denoiser(xt, tt), noise)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch % max(1, args.log_every) == 0:
            print(f"direct_diff_epoch={epoch:03d} loss={np.mean(losses):.6f}")
    denoiser.eval()
    fake = ddpm.sample_sequence(denoiser, len(test_data), args.seq_len, args.channels).detach().cpu().numpy()
    metrics = {
        "discriminative": discriminative_score(test_data, fake, device, args.disc_epochs),
        "mmd": rbf_mmd(test_data, fake),
        "acf_error": acf_error(test_data, fake),
        "marginal_error": marginal_error(test_data, fake),
    }
    return fake, metrics, denoiser


def train_itspm_conditioned_diffusion(
    args: argparse.Namespace,
    cfg: ITSPMConfig,
    data: np.ndarray,
    obs_data: np.ndarray,
    mask: np.ndarray,
    times: np.ndarray,
    train_data: np.ndarray,
    test_data: np.ndarray,
    split: int,
    device: torch.device,
) -> tuple[np.ndarray, dict, nn.Module]:
    ddpm = SequenceDDPM(args.diffusion_steps, args.beta_start, args.beta_end, device)
    denoiser = ITSPMConditionedDenoiser(cfg, args.channels, args.diffusion_hidden, args.diffusion_steps).to(device)
    opt = torch.optim.AdamW(denoiser.parameters(), lr=args.diff_lr, weight_decay=args.weight_decay)
    x_all = torch.from_numpy(data.astype(np.float32)).to(device)
    obs_all = torch.from_numpy(obs_data.astype(np.float32)).to(device)
    mask_all = torch.from_numpy(mask.astype(np.float32)).to(device)
    times_all = torch.from_numpy(times.astype(np.float32)).to(device)
    for epoch in range(1, args.diff_epochs + 1):
        losses = []
        denoiser.train()
        for ids_np in iter_batches(split, args.batch_size, True, args.seed + 2000 + epoch):
            ids = torch.as_tensor(ids_np, device=device)
            x0 = x_all[ids]
            tt = torch.randint(0, args.diffusion_steps, (x0.size(0),), device=device)
            noise = torch.randn_like(x0)
            xt = ddpm.q_sample(x0, tt, noise)
            opt.zero_grad(set_to_none=True)
            pred_noise = denoiser(xt, tt, obs_all[ids], mask_all[ids], times_all[ids])
            loss = F.mse_loss(pred_noise, noise)
            if args.x0_loss_weight > 0 or args.marginal_loss_weight > 0:
                x0_pred = ddpm.predict_x0(xt, tt, pred_noise)
                if args.x0_loss_weight > 0:
                    loss = loss + args.x0_loss_weight * F.mse_loss(x0_pred, x0)
                if args.marginal_loss_weight > 0:
                    pred_mean = x0_pred.mean(dim=(0, 1))
                    true_mean = x0.mean(dim=(0, 1))
                    pred_std = x0_pred.std(dim=(0, 1))
                    true_std = x0.std(dim=(0, 1))
                    marginal_loss = F.mse_loss(pred_mean, true_mean) + F.mse_loss(pred_std, true_std)
                    loss = loss + args.marginal_loss_weight * marginal_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch % max(1, args.log_every) == 0:
            print(f"itspm_cond_diff_epoch={epoch:03d} loss={np.mean(losses):.6f}")

    denoiser.eval()
    test_slice = slice(split, len(data))
    with torch.no_grad():
        fake = ddpm.sample_conditioned(
            denoiser,
            obs_all[test_slice],
            mask_all[test_slice],
            times_all[test_slice],
            args.clamp_observed,
        ).detach().cpu().numpy()
    metrics = {
        "discriminative": discriminative_score(test_data, fake, device, args.disc_epochs),
        "mmd": rbf_mmd(test_data, fake),
        "acf_error": acf_error(test_data, fake),
        "marginal_error": marginal_error(test_data, fake),
        "observed_mse": float(((fake - test_data) ** 2 * mask[split:]).sum() / (mask[split:].sum() + 1e-6)),
        "missing_mse": float(((fake - test_data) ** 2 * (1.0 - mask[split:])).sum() / ((1.0 - mask[split:]).sum() + 1e-6)),
    }
    return fake, metrics, denoiser




class Discriminator(nn.Module):
    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.gru = nn.GRU(channels, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        return self.head(h[-1]).squeeze(-1)


def rbf_mmd(x: np.ndarray, y: np.ndarray, max_samples: int = 512) -> float:
    rng = np.random.default_rng(0)
    if len(x) > max_samples:
        x = x[rng.choice(len(x), max_samples, replace=False)]
    if len(y) > max_samples:
        y = y[rng.choice(len(y), max_samples, replace=False)]
    x = x.reshape(len(x), -1)
    y = y.reshape(len(y), -1)
    xy = np.vstack([x, y])
    sq = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
    sigma = np.sqrt(np.median(sq[sq > 0]) + 1e-6)
    gamma = 1.0 / (2.0 * sigma * sigma)
    kxx = np.exp(-gamma * ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)).mean()
    kyy = np.exp(-gamma * ((y[:, None, :] - y[None, :, :]) ** 2).sum(-1)).mean()
    kxy = np.exp(-gamma * ((x[:, None, :] - y[None, :, :]) ** 2).sum(-1)).mean()
    return float(kxx + kyy - 2.0 * kxy)


def flat_kl(real: np.ndarray, fake: np.ndarray) -> float:
    hist_real, edge_real = np.histogram(real[~np.isnan(real)], density=True, bins=50)
    hist_fake, _ = np.histogram(fake[~np.isnan(fake)], density=True, bins=edge_real)
    return float(entropy(hist_real, hist_fake + 1e-9))


def official_mdd(real: np.ndarray, fake: np.ndarray, n_bins: int = 20) -> float:
    x_real = torch.as_tensor(real, dtype=torch.float32)
    x_fake = torch.as_tensor(fake, dtype=torch.float32)
    losses = []
    for c in range(x_real.shape[2]):
        for t in range(x_real.shape[1]):
            xr = x_real[:, t, c]
            xf = x_fake[:, t, c]
            a = xr.min().item()
            b = xr.max().item()
            if b == a:
                b += 1e-2
            bins = torch.linspace(a, b, n_bins + 1)
            delta = bins[1] - bins[0]
            real_density = torch.histc(xr, bins=n_bins, min=a, max=b).float()
            fake_density = torch.histc(xf, bins=n_bins, min=a, max=b).float()
            real_density = real_density / delta / float(xr.numel())
            fake_density = fake_density / delta / float(xf.numel())
            losses.append(torch.abs(fake_density - real_density).mean())
    return float(torch.stack(losses).mean().item())


def acf_error(real: np.ndarray, fake: np.ndarray, max_lag: int = 8) -> float:
    errs = []
    for lag in range(1, max_lag + 1):
        r = (real[:, :-lag] * real[:, lag:]).mean(axis=(0, 1))
        f = (fake[:, :-lag] * fake[:, lag:]).mean(axis=(0, 1))
        errs.append(np.abs(r - f).mean())
    return float(np.mean(errs))


def marginal_error(real: np.ndarray, fake: np.ndarray) -> float:
    return float(np.abs(real.mean(axis=(0, 1)) - fake.mean(axis=(0, 1))).mean() +
                 np.abs(real.std(axis=(0, 1)) - fake.std(axis=(0, 1))).mean())


def discriminative_score(real: np.ndarray, fake: np.ndarray, device: torch.device, epochs: int = 20) -> float:
    n = min(len(real), len(fake))
    x = np.concatenate([real[:n], fake[:n]], axis=0).astype(np.float32)
    y = np.concatenate([np.ones(n), np.zeros(n)], axis=0).astype(np.float32)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(x))
    split = int(0.7 * len(x))
    tr, te = idx[:split], idx[split:]
    model = Discriminator(real.shape[-1], 32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    xb = torch.from_numpy(x).to(device)
    yb = torch.from_numpy(y).to(device)
    for _ in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(xb[tr]), yb[tr])
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred = (torch.sigmoid(model(xb[te])) > 0.5).detach().cpu().numpy().astype(np.float32)
    return float(abs(accuracy_score(y[te], pred) - 0.5))


def table1_discriminative_score(
    real: np.ndarray,
    fake: np.ndarray,
    device: torch.device,
    iterations: int,
    batch_size: int,
    hidden_dim: int | None = None,
) -> float:
    real_t = torch.as_tensor(real, dtype=torch.float32, device=device)
    fake_t = torch.as_tensor(fake, dtype=torch.float32, device=device)
    if hidden_dim is None:
        hidden_dim = max(4, real.shape[-1] // 2)

    class RNNDiscriminator(nn.Module):
        def __init__(self, inp_dim: int, hidden: int):
            super().__init__()
            self.rnn = nn.GRU(inp_dim, hidden, batch_first=True)
            self.linear = nn.Linear(hidden, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            _, h = self.rnn(x)
            return self.linear(h[-1]).squeeze(-1)

    rng = np.random.default_rng(0)
    idx_real = rng.permutation(len(real))
    idx_fake = rng.permutation(len(fake))
    split_real = int(0.8 * len(real))
    split_fake = int(0.8 * len(fake))
    train_real = torch.as_tensor(idx_real[:split_real], dtype=torch.long, device=device)
    test_real = torch.as_tensor(idx_real[split_real:], dtype=torch.long, device=device)
    train_fake = torch.as_tensor(idx_fake[:split_fake], dtype=torch.long, device=device)
    test_fake = torch.as_tensor(idx_fake[split_fake:], dtype=torch.long, device=device)
    model = RNNDiscriminator(real.shape[-1], hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters())
    for _ in range(iterations):
        ridx = train_real[torch.randint(0, len(train_real), (batch_size,), device=device)]
        fidx = train_fake[torch.randint(0, len(train_fake), (batch_size,), device=device)]
        logits_real = model(real_t[ridx])
        logits_fake = model(fake_t[fidx])
        loss = F.binary_cross_entropy_with_logits(logits_real, torch.ones_like(logits_real))
        loss = loss + F.binary_cross_entropy_with_logits(logits_fake, torch.zeros_like(logits_fake))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(real_t[test_real]), model(fake_t[test_fake])], dim=0)
        labels = torch.cat([torch.ones(len(test_real), device=device), torch.zeros(len(test_fake), device=device)])
        pred = (torch.sigmoid(logits) > 0.5).float()
        acc = (pred == labels).float().mean().item()
    return float(abs(0.5 - acc))


def table1_metrics(real: np.ndarray, fake: np.ndarray, device: torch.device, args: argparse.Namespace) -> dict:
    ds = table1_discriminative_score(real, fake, device, args.ds_iterations, args.ds_batch_size)
    return {
        "ds": ds,
        "mdd": official_mdd(real, fake, args.mdd_bins),
        "kl": flat_kl(real, fake),
        "mmd": rbf_mmd(real, fake),
        "acf_error": acf_error(real, fake),
    }


def baseline_gaussian(train: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    flat = train.reshape(len(train), -1)
    mu = flat.mean(axis=0)
    std = flat.std(axis=0) + 1e-4
    return rng.normal(mu, std, size=(n_samples, flat.shape[1])).astype(np.float32).reshape(n_samples, *train.shape[1:])


def calibrate_marginals(fake: np.ndarray, train: np.ndarray) -> np.ndarray:
    fake_mean = fake.mean(axis=(0, 1), keepdims=True)
    fake_std = fake.std(axis=(0, 1), keepdims=True) + 1e-6
    train_mean = train.mean(axis=(0, 1), keepdims=True)
    train_std = train.std(axis=(0, 1), keepdims=True) + 1e-6
    return ((fake - fake_mean) / fake_std * train_std + train_mean).astype(np.float32)


def calibrate_missing_marginals(fake: np.ndarray, train: np.ndarray, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    calibrated = fake.copy()
    train_mean = train.mean(axis=(0, 1))
    train_std = train.std(axis=(0, 1)) + 1e-6
    for c in range(fake.shape[-1]):
        missing = mask[:, :, c] < 0.5
        if not np.any(missing):
            continue
        values = calibrated[:, :, c][missing]
        calibrated[:, :, c][missing] = (values - values.mean()) / (values.std() + 1e-6) * train_std[c] + train_mean[c]
    return (calibrated * (1.0 - mask) + obs * mask).astype(np.float32)


def iter_batches(n: int, batch_size: int, shuffle: bool, seed: int):
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    for start in range(0, n, batch_size):
        yield idx[start:start + batch_size]


def train(args: argparse.Namespace) -> dict:
    start = time.time()
    if args.gpu != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu != "cpu" else "cpu")

    data, mask, is_table1_dataset = load_table1_dataset(args)
    args.channels = data.shape[-1]
    data, mean, std, train_ids = normalize_train_test(data, args.train_frac)
    split = len(train_ids)
    train_data = data[:split]
    test_data = data[split:]
    obs_data = data * mask
    times = np.broadcast_to(np.linspace(0.0, 1.0, args.seq_len, dtype=np.float32), data.shape[:2]).copy()

    table1_dataset = "sines" if args.dataset == "sine" else args.dataset
    missing_key = round(float(args.missing), 1)
    diffmn_ref = {
        metric: DIFF_MN_TABLE1[metric].get(missing_key, {}).get(table1_dataset)
        for metric in ("ds", "mdd", "kl")
    }
    train_eval = denormalize(train_data, mean, std)
    test_eval = denormalize(test_data, mean, std)
    gauss = baseline_gaussian(train_data, len(test_data), args.seed + 99)
    gauss_eval = denormalize(gauss, mean, std)
    metrics = {
        "gaussian_baseline": table1_metrics(test_eval, gauss_eval, device, args),
    }
    out_dir = TASK_DIR / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ITSPMConfig(
        input_dim=args.channels,
        d_model=args.d_model,
        dropout=args.dropout,
        n_ref_points=args.n_ref_points,
        n_scales=args.n_scales,
        n_mixer_layers=args.n_mixer_layers,
        max_event_tokens=args.max_event_tokens,
        max_gap_tokens=args.max_gap_tokens,
    )

    if args.generator == "direct_diffusion":
        fake, direct_metrics, denoiser = train_direct_diffusion(args, train_data, test_data, device)
        if args.calibrate_marginals:
            fake = calibrate_marginals(fake, train_data)
            fake_eval = denormalize(fake, mean, std)
            direct_metrics = table1_metrics(test_eval, fake_eval, device, args)
        else:
            fake_eval = denormalize(fake, mean, std)
            direct_metrics = table1_metrics(test_eval, fake_eval, device, args)
        metrics["direct_sequence_diffusion"] = direct_metrics
        np.save(out_dir / "generated.npy", fake_eval.astype(np.float32))
        np.save(out_dir / "real_test.npy", test_eval.astype(np.float32))
        torch.save({"denoiser": denoiser.state_dict()}, out_dir / "model.pt")
        result = {
            "metrics": metrics,
            "diffmn_table1_reference": diffmn_ref,
            "table1_dataset_available": is_table1_dataset,
            "args": vars(args),
            "elapsed_sec": time.time() - start,
            "source": "Task-local direct sequence diffusion upper-bound inspired by TimeCraft Diff-MN.",
            "paper_protocol": "upper_bound_not_itspm",
        }
        with open(out_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return result

    if args.generator == "itspm_conditioned_diffusion":
        fake, cond_metrics, denoiser = train_itspm_conditioned_diffusion(
            args, cfg, data, obs_data, mask, times, train_data, test_data, split, device)
        if args.calibrate_marginals:
            if args.clamp_observed:
                fake = calibrate_missing_marginals(fake, train_data, obs_data[split:], mask[split:])
            else:
                fake = calibrate_marginals(fake, train_data)
        fake_eval = denormalize(fake, mean, std)
        obs_eval = denormalize(obs_data[split:], mean, std)
        cond_metrics = table1_metrics(test_eval, fake_eval, device, args)
        cond_metrics["observed_mse"] = float(((fake - test_data) ** 2 * mask[split:]).sum() / (mask[split:].sum() + 1e-6))
        cond_metrics["missing_mse"] = float(((fake - test_data) ** 2 * (1.0 - mask[split:])).sum() / ((1.0 - mask[split:]).sum() + 1e-6))
        metrics["itspm_conditioned_diffusion"] = cond_metrics
        np.save(out_dir / "generated.npy", fake_eval.astype(np.float32))
        np.save(out_dir / "real_test.npy", test_eval.astype(np.float32))
        np.save(out_dir / "observed_test.npy", obs_eval.astype(np.float32))
        np.save(out_dir / "mask_test.npy", mask[split:].astype(np.float32))
        torch.save({"denoiser": denoiser.state_dict()}, out_dir / "model.pt")
        result = {
            "metrics": metrics,
            "diffmn_table1_reference": diffmn_ref,
            "table1_dataset_available": is_table1_dataset,
            "args": vars(args),
            "itspm_config": asdict(cfg),
            "elapsed_sec": time.time() - start,
            "source": (
                "One-for-all ITSPM backbone from parent models/ITSPM.py plus a "
                "generation task head. The ITSPM architecture is not modified."
            ),
            "paper_protocol": "one_for_all_backbone",
            "postprocessing": {
                "marginal_calibration": bool(args.calibrate_marginals),
                "observed_value_clamping": bool(args.clamp_observed),
            },
        }
        with open(out_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return result

    ae = ITSPMLatentAutoencoder(cfg, args.seq_len, args.channels, args.latent_dim).to(device)
    opt_ae = torch.optim.AdamW(ae.parameters(), lr=args.ae_lr, weight_decay=args.weight_decay)

    x_all = torch.from_numpy(obs_data).to(device)
    m_all = torch.from_numpy(mask).to(device)
    t_all = torch.from_numpy(times).to(device)
    y_all = torch.from_numpy(data).to(device)

    for epoch in range(1, args.ae_epochs + 1):
        losses = []
        ae.train()
        for ids_np in iter_batches(split, args.batch_size, True, args.seed + epoch):
            ids = torch.as_tensor(ids_np, device=device)
            opt_ae.zero_grad(set_to_none=True)
            recon, _ = ae(t_all[ids], x_all[ids], m_all[ids])
            loss_obs = ((recon - y_all[ids]).pow(2) * m_all[ids]).sum() / (m_all[ids].sum() + 1e-6)
            loss_full = F.mse_loss(recon, y_all[ids])
            loss = loss_obs + args.full_recon_weight * loss_full
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
            opt_ae.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch % max(1, args.log_every) == 0:
            print(f"ae_epoch={epoch:03d} loss={np.mean(losses):.6f}")

    ae.eval()
    latents = []
    with torch.no_grad():
        for ids_np in iter_batches(split, args.batch_size, False, args.seed):
            ids = torch.as_tensor(ids_np, device=device)
            latents.append(ae.encode(t_all[ids], x_all[ids], m_all[ids]).detach().cpu())
    z_train = torch.cat(latents, dim=0).to(device)
    z_mean = z_train.mean(0, keepdim=True)
    z_std = z_train.std(0, keepdim=True) + 1e-6
    z_norm = (z_train - z_mean) / z_std

    ddpm = LatentDDPM(args.diffusion_steps, args.beta_start, args.beta_end, device)
    denoiser = LatentDenoiser(args.latent_dim, args.diffusion_hidden, args.diffusion_steps).to(device)
    opt_diff = torch.optim.AdamW(denoiser.parameters(), lr=args.diff_lr, weight_decay=args.weight_decay)
    for epoch in range(1, args.diff_epochs + 1):
        losses = []
        denoiser.train()
        for ids_np in iter_batches(len(z_norm), args.batch_size, True, args.seed + 1000 + epoch):
            z0 = z_norm[torch.as_tensor(ids_np, device=device)]
            tt = torch.randint(0, args.diffusion_steps, (z0.size(0),), device=device)
            noise = torch.randn_like(z0)
            zt = ddpm.q_sample(z0, tt, noise)
            opt_diff.zero_grad(set_to_none=True)
            loss = F.mse_loss(denoiser(zt, tt), noise)
            loss.backward()
            opt_diff.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch % max(1, args.log_every) == 0:
            print(f"diff_epoch={epoch:03d} loss={np.mean(losses):.6f}")

    denoiser.eval()
    ae.eval()
    with torch.no_grad():
        z_fake = ddpm.sample(denoiser, len(test_data), args.latent_dim) * z_std + z_mean
        fake = ae.decode(z_fake).detach().cpu().numpy()
    if args.calibrate_marginals:
        fake = calibrate_marginals(fake, train_data)
    fake_eval = denormalize(fake, mean, std)

    metrics.update({
        "itspm_latent_diffusion": table1_metrics(test_eval, fake_eval, device, args)
    })

    np.save(out_dir / "generated.npy", fake_eval.astype(np.float32))
    np.save(out_dir / "real_test.npy", test_eval.astype(np.float32))
    torch.save({"autoencoder": ae.state_dict(), "denoiser": denoiser.state_dict()}, out_dir / "model.pt")
    result = {
        "metrics": metrics,
        "diffmn_table1_reference": diffmn_ref,
        "table1_dataset_available": is_table1_dataset,
        "args": vars(args),
        "itspm_config": asdict(cfg),
        "elapsed_sec": time.time() - start,
        "source": (
            "One-for-all ITSPM backbone from parent models/ITSPM.py plus a "
            "latent generation task head. The ITSPM architecture is not modified."
        ),
        "paper_protocol": "one_for_all_backbone",
        "postprocessing": {
            "marginal_calibration": bool(args.calibrate_marginals),
            "observed_value_clamping": False,
        },
    }
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Irregular time series generation with ITSPM latent diffusion")
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--dataset", type=str, default="sines",
                   choices=["sine", "sines", "stocks", "energy", "mujoco", "polynomial"])
    p.add_argument("--generator", type=str, default="itspm_conditioned_diffusion",
                   choices=["itspm_latent_diffusion", "direct_diffusion", "itspm_conditioned_diffusion"])
    p.add_argument("--n_samples", type=int, default=1024)
    p.add_argument("--data_root", type=str, default=str(TASK_DIR / "table1_data"))
    p.add_argument("--seq_len", type=int, default=36)
    p.add_argument("--channels", type=int, default=5)
    p.add_argument("--missing", type=float, default=0.5)
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--ae_epochs", type=int, default=40)
    p.add_argument("--diff_epochs", type=int, default=80)
    p.add_argument("--disc_epochs", type=int, default=20)
    p.add_argument("--ds_iterations", type=int, default=2000)
    p.add_argument("--ds_batch_size", type=int, default=128)
    p.add_argument("--mdd_bins", type=int, default=20)
    p.add_argument("--ae_lr", type=float, default=1e-3)
    p.add_argument("--diff_lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--full_recon_weight", type=float, default=0.25)
    p.add_argument("--x0_loss_weight", type=float, default=0.0,
                   help="Optional training-time x0 reconstruction loss for the generation head.")
    p.add_argument("--marginal_loss_weight", type=float, default=0.0,
                   help="Optional training-time batch mean/std matching loss for the generation head.")
    p.add_argument("--latent_dim", type=int, default=32)
    p.add_argument("--diffusion_steps", type=int, default=50)
    p.add_argument("--diffusion_hidden", type=int, default=128)
    p.add_argument("--calibrate_marginals", action="store_true",
                   help="Optional post-hoc marginal calibration. Use only as an ablation, not as the default paper protocol.")
    p.add_argument("--clamp_observed", action=argparse.BooleanOptionalAction, default=False,
                   help="Optionally preserve observed values during conditional sampling. Disabled by default for one-for-all reporting.")
    p.add_argument("--beta_start", type=float, default=1e-4)
    p.add_argument("--beta_end", type=float, default=0.02)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--n_ref_points", type=int, default=24)
    p.add_argument("--n_scales", type=int, default=2)
    p.add_argument("--n_mixer_layers", type=int, default=1)
    p.add_argument("--max_event_tokens", type=int, default=48)
    p.add_argument("--max_gap_tokens", type=int, default=24)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--out_dir", type=str, default="outputs/default")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
