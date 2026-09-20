"""Train/reload the original MuJoCo conditioned DDPM and measure backbone CKA."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.cka.common import ITSPMRepresentationCapture, deterministic_indices, save_cka_result


def load_generation_module():
    path = ROOT / 'Generation/irregular_generation_diffmn/train_irregular_generation.py'
    spec = importlib.util.spec_from_file_location('itspm_original_generation', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seq-len', type=int, choices=[12, 24, 36], default=36)
    p.add_argument('--missing', type=float, default=0.5)
    p.add_argument('--n-samples', type=int, default=10000)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--max-samples', type=int, default=2048)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--output-dir', type=Path, default=ROOT / 'experiments/cka/outputs/generation_mujoco')
    p.add_argument('--data-root', type=Path, default=ROOT / 'Generation/irregular_generation_diffmn/table1_data')
    p.add_argument('--no-resume', action='store_true')
    return p.parse_args()


def main():
    a = parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    device = torch.device(a.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('A Colab CUDA GPU is required. CPU is allowed only when explicitly requested for a smoke test.')
    if not 0 <= a.missing < 1 or a.epochs < 1 or a.batch_size < 1 or a.max_samples < 2:
        raise ValueError('Expected 0 <= missing < 1, epochs/batch_size >= 1, and max_samples >= 2.')
    source = a.data_root / f'mujoco_training_{a.seq_len}.pt'
    if not source.exists():
        raise FileNotFoundError(f'{source} is absent. Run prepare_mujoco.py or provide the original file. Proxy fallback is disabled.')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    g = load_generation_module()
    g.set_seed(a.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    raw, mask, is_real = g.load_table1_dataset(argparse.Namespace(
        dataset='mujoco', data_root=str(a.data_root), seq_len=a.seq_len,
        n_samples=a.n_samples, missing=a.missing, seed=a.seed, channels=14))
    if not is_real or raw.ndim != 3 or raw.shape[1:] != (a.seq_len, 14):
        raise ValueError('Expected a supplied MuJoCo data file shaped [N,L,14]; proxy fallback is disabled.')
    if not np.isfinite(raw).all():
        raise ValueError('Dataset contains non-finite values.')
    data, mean, std, train_ids = g.normalize_train_test(raw, 0.8)
    split = len(train_ids)
    if split < 1 or len(data) - split < 2:
        raise ValueError('Need at least one training trajectory and two held-out trajectories for CKA.')
    obs = data * mask
    times = np.broadcast_to(np.linspace(0., 1., a.seq_len, dtype=np.float32), data.shape[:2]).copy()
    cfg = g.ITSPMConfig(input_dim=14, d_model=96, dropout=0.1, n_ref_points=32,
                        n_scales=2, n_mixer_layers=1, max_event_tokens=64, max_gap_tokens=32)
    model = g.ITSPMConditionedDenoiser(cfg, 14, 256, 100).to(device)
    ddpm = g.SequenceDDPM(100, 1e-4, 0.02, device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    # Arrays stay on host, transferring only minibatches to make GPU memory bounded.
    arrays = [torch.from_numpy(x.astype(np.float32)) for x in (data, obs, mask, times)]
    model_path = a.output_dir / 'model.pt'
    data_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    run_config = {
        'task': 'generation', 'dataset': 'mujoco', 'seed': a.seed, 'seq_len': a.seq_len,
        'missing': a.missing, 'n_samples_actual': len(data), 'train_size': split,
        'test_size': len(data) - split, 'itspm_config': asdict(cfg),
        'diffusion_hidden': 256, 'diffusion_steps': 100, 'beta_start': 1e-4,
        'beta_end': 0.02, 'learning_rate': 1e-3, 'weight_decay': 1e-5,
        'x0_loss_weight': 0.05, 'marginal_loss_weight': 0., 'batch_size': a.batch_size,
        'data_sha256': data_sha, 'requested_epochs': a.epochs,
    }
    start_epoch = 0
    checkpoint_training_config = None
    externally_loaded = a.checkpoint is not None
    restore = a.checkpoint or (model_path if model_path.exists() and not a.no_resume else None)
    if restore:
        checkpoint = torch.load(restore, map_location='cpu', weights_only=False)
        checkpoint_training_config = checkpoint.get('run_config')
        if not externally_loaded:
            old_cfg = checkpoint['run_config']
            for key, val in run_config.items():
                if key != 'requested_epochs' and old_cfg.get(key) != val:
                    raise ValueError(f'Existing run differs at {key}; use another output directory.')
        elif checkpoint_training_config is not None:
            # Inference batching/selection seed may change, but the model must
            # see the same preprocessing, observed condition and architecture.
            for key in ('seq_len', 'missing', 'n_samples_actual', 'train_size',
                        'itspm_config', 'diffusion_hidden', 'diffusion_steps', 'data_sha256'):
                if checkpoint_training_config.get(key) != run_config[key]:
                    raise ValueError(f'Checkpoint/data configuration differs at {key}.')
        model.load_state_dict(checkpoint.get('denoiser', checkpoint), strict=True)
        start_epoch = checkpoint.get('epoch', 0)
        if not externally_loaded and 'optimizer' in checkpoint:
            opt.load_state_dict(checkpoint['optimizer'])
            torch.set_rng_state(checkpoint['torch_rng'])
            if device.type == 'cuda' and checkpoint.get('cuda_rng') is not None:
                torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        print(f'Loaded {restore}; recorded epoch={start_epoch}', flush=True)
    (a.output_dir / 'config.json').write_text(json.dumps(run_config, indent=2) + '\n')
    if not externally_loaded:
        if start_epoch == 0 and (a.output_dir / 'training.jsonl').exists():
            (a.output_dir / 'training.jsonl').unlink()
        begun = time.monotonic()
        for epoch in range(start_epoch + 1, a.epochs + 1):
            model.train()
            loss_sum, count = 0., 0
            for indices in g.iter_batches(split, a.batch_size, True, a.seed + 2000 + epoch):
                x0, o, m, ts = [x[indices].to(device) for x in arrays]
                tt = torch.randint(0, 100, (len(indices),), device=device)
                noise = torch.randn_like(x0)
                xt = ddpm.q_sample(x0, tt, noise)
                opt.zero_grad(set_to_none=True)
                eps = model(xt, tt, o, m, ts)
                loss = F.mse_loss(eps, noise) + 0.05 * F.mse_loss(ddpm.predict_x0(xt, tt, eps), x0)
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Non-finite training loss at epoch {epoch}.')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                opt.step()
                loss_sum += float(loss.detach()) * len(indices)
                count += len(indices)
            record = {'epoch': epoch, 'train_loss': loss_sum / count,
                      'elapsed_seconds': time.monotonic() - begun}
            print(json.dumps(record), flush=True)
            with (a.output_dir / 'training.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            # Persist each epoch so a Colab runtime interruption loses <1 epoch.
            checkpoint = {'denoiser': model.state_dict(), 'optimizer': opt.state_dict(),
                          'epoch': epoch, 'run_config': run_config,
                          'torch_rng': torch.get_rng_state(),
                          'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None}
            temporary = model_path.with_suffix('.tmp')
            torch.save(checkpoint, temporary)
            temporary.replace(model_path)
        checkpoint_path = model_path
        completed_epochs = max(start_epoch, a.epochs)
    else:
        checkpoint_path = a.checkpoint
        completed_epochs = start_epoch or None
    model.eval()
    ids = deterministic_indices(len(data) - split, a.max_samples, a.seed + 31415) + split
    firsts, finals, shareds, valid = [], [], [], []
    with torch.inference_mode(), ITSPMRepresentationCapture(model.itspm) as capture:
        for start in range(0, len(ids), a.batch_size):
            ix = ids[start:start + a.batch_size]
            o, m, ts = [x[ix].to(device) for x in arrays[1:]]
            capture.begin_batch(m)
            model.itspm(ts, o, m)
            pair = capture.end_batch(last='forward')
            firsts.append(pair['first'])
            finals.append(pair['last'])
            valid.append(pair['valid'])
            shareds.append(pair['shared_global'])
    first, final = torch.cat(firsts), torch.cat(finals)
    common_meta = {
        **run_config, 'trained_epochs': completed_epochs, 'checkpoint': str(checkpoint_path),
        'checkpoint_sha256': hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        'device': str(device), 'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'torch_version': torch.__version__, 'split': 'test', 'row_unit': 'held-out trajectory',
        'first_layer': 'itspm.encoder.norm output, masked mean across observed time x channel',
        'last_layer': 'itspm.forward output = shared global + cls_fusion; input to cond_proj',
        'checkpoint_selection': ('user-supplied checkpoint; selection procedure not verified'
                                 if externally_loaded else 'fixed final epoch (no test-based selection)'),
        'training_protocol': ('not run by this command; see checkpoint_training_config if available'
                              if externally_loaded else 'Generation/scripts/mujoco.sh hyperparameters, one length/missing setting; full 80% training split; original noise+x0 loss'),
        'checkpoint_training_config': checkpoint_training_config if externally_loaded else run_config,
        'training_metadata_available': not externally_loaded or checkpoint_training_config is not None,
        'representation_protocol': 'fixed observed condition, independent of diffusion step/noise; native ITSPM forward in eval mode',
        'normalization': 'original loader minmax across loaded trajectories, then train-only channel z-score',
        'split_protocol': 'first floor(0.8*N) source-file trajectories train; remaining trajectories held out; no shuffle',
        'mask_protocol': 'original full-timestep missingness; mask seed=56789; first and last timesteps restored',
        'observed_fraction_test': float(mask[split:].mean()),
        'cka_selection_seed': a.seed + 31415,
        'cka_max_samples_requested': a.max_samples,
        'deterministic_algorithms': True,
        'is_smoke_test': device.type != 'cuda' or a.epochs < 200 or a.n_samples < 10000,
    }
    provenance = source.with_suffix('.provenance.json')
    common_meta['data_provenance'] = json.loads(provenance.read_text()) if provenance.exists() else {'provenance': 'provided_dataset', 'path': str(source)}
    if common_meta['data_provenance'].get('provenance') == 'synthetic_smoke_fixture':
        common_meta['is_smoke_test'] = True
        common_meta['dataset'] = 'synthetic_14d_smoke_fixture_not_mujoco'
    report = save_cka_result(a.output_dir, first, final, ids, common_meta, valid=torch.cat(valid))
    save_cka_result(a.output_dir / 'shared_global_endpoint', first, torch.cat(shareds), ids,
                    {**common_meta, 'last_layer': 'interaction global + kernel_branch global (before cls_fusion)'}, valid=torch.cat(valid))
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == '__main__':
    main()
