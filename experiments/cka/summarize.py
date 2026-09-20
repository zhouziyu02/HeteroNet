"""Build a table and static figure from completed, real GPU CKA results only."""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TASKS = [('interpolation_activity', 'Interpolation', 'Activity'),
         ('extrapolation_activity', 'Extrapolation', 'Activity'),
         ('tpp_taxi', 'TPP', 'Taxi'), ('generation_mujoco', 'Generation', 'MuJoCo')]


def validate_result(path, task, dataset, expected_seed=None):
    """Verify provenance fields and independently recompute paired-array CKA."""
    from experiments.cka.common import linear_cka
    path = Path(path)
    result = json.loads(path.read_text())
    meta = result.get('metadata', {})
    if result.get('status') != 'ok':
        raise ValueError(f'{path}: CKA is not defined: {result.get("reason")}')
    if meta.get('smoke_test') or meta.get('is_smoke_test'):
        raise ValueError(f'{path}: smoke tests are not research results')
    device = str(meta.get('device', ''))
    if device != 'cuda' and not device.startswith('cuda:'):
        raise ValueError(f'{path}: result does not record CUDA execution')
    if meta.get('task', '').lower() != task.lower() or meta.get('dataset', '').lower() != dataset.lower():
        raise ValueError(f'{path}: task/dataset metadata do not match its result directory')
    seed = meta.get('seed')
    if seed is None or (expected_seed is not None and seed != expected_seed):
        raise ValueError(f'{path}: missing or inconsistent experimental seed')
    # A training seed may differ only for an explicitly analyzed external
    # checkpoint. The suite does not request such cross-seed evaluations.
    if expected_seed is not None and meta.get('training_seed', seed) != expected_seed:
        raise ValueError(f'{path}: training seed does not match the requested suite seed')
    if result.get('representation_file') != 'representations.npz':
        raise ValueError(f'{path}: missing expected paired representation artifact')
    with np.load(path.parent / 'representations.npz', allow_pickle=False) as rep:
        first, last, ids = rep['first'], rep['last'], rep['sample_ids']
        count = len(first)
        if first.ndim != 2 or last.ndim != 2 or len(last) != count:
            raise ValueError(f'{path}: invalid paired representation shapes')
        if result.get('n_samples') != count or ids.shape != (count,):
            raise ValueError(f'{path}: saved paired row counts differ from the report')
        if result.get('first_features') != first.shape[1] or result.get('last_features') != last.shape[1]:
            raise ValueError(f'{path}: saved feature counts differ from the report')
        saved_ids = list(map(str, ids.tolist()))
        if saved_ids != list(map(str, result.get('sample_ids', []))) or len(set(saved_ids)) != count:
            raise ValueError(f'{path}: paired sample IDs are missing, duplicated, or reordered')
        cka = linear_cka(first, last)
    if not np.isclose(cka, result.get('linear_cka'), rtol=1e-10, atol=1e-12):
        raise RuntimeError(f'Saved representations do not reproduce {path}.')
    return result, cka


def summarize(output_dir, seed=None):
    output_dir = Path(output_dir)
    rows, completed = [], []
    for directory, task, dataset in TASKS:
        path = output_dir / directory / 'results.json'
        if not path.exists():
            rows.append(f'| {task} | {dataset} | — | — | — | not completed |')
            continue
        r = json.loads(path.read_text())
        meta = r.get('metadata', {})
        if meta.get('smoke_test') or meta.get('is_smoke_test'):
            rows.append(f'| {task} | {dataset} | — | — | — | smoke test，not a formal result |')
            continue
        device = str(meta.get('device', ''))
        if device != 'cuda' and not device.startswith('cuda:'):
            rows.append(f'| {task} | {dataset} | — | — | — | CUDA execution not recorded，not a formal result |')
            continue
        if r['status'] != 'ok':
            rows.append(f'| {task} | {dataset} | — | {r["n_samples"]} | — | undefined CKA |')
            continue
        r, cka = validate_result(path, task, dataset, seed)
        rows.append(f'| {task} | {dataset} | {cka:.6f} | {r["n_samples"]} | {meta["seed"]} | completed |')
        completed.append({'task': task, 'dataset': dataset, 'linear_cka': cka, 'n_samples': r['n_samples'],
                          'seed': meta['seed'], 'result_path': path.relative_to(output_dir).as_posix()})
    text = ('# CKA: early event and final backbone representations\n\n'
            'Centered linear CKA (Kornblith et al., ICML 2019), globally centered over paired held-out samples.\n\n'
            '| Task | Dataset | Linear CKA | Paired samples | Seed | Status |\n|---|---|---:|---:|---:|---|\n' + '\n'.join(rows) + '\n\n'
            'First endpoint: EventEncoder output, masked mean over observed events per sample. '
            'Last endpoint: the final backbone representation consumed by the task head. See each results.json for exact endpoints.\n\n'
            'Rows are held-out windows (Activity), next-event contexts (Taxi), or trajectories (MuJoCo). '
            'These are descriptive similarities from one training seed, not predictive accuracy or a mean across seeds.\n')
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'CKA_RESULTS.md').write_text(text)
    (output_dir / 'summary.json').write_text(json.dumps(completed, indent=2) + '\n')
    if completed:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        labels = [f'{r["task"]}\n{r["dataset"]}' for r in completed]
        bars = ax.bar(labels, [r['linear_cka'] for r in completed], color=['#235789', '#2C8C99', '#D18B30', '#8661A1'][:len(completed)], width=.58)
        ax.set(ylim=(0, 1.08), ylabel='Centered linear CKA', title='ITSPM: early event vs final backbone representation')
        ax.spines[['right', 'top']].set_visible(False)
        ax.bar_label(bars, fmt='%.3f', padding=5)
        fig.tight_layout()
        fig.savefig(output_dir / 'CKA_RESULTS.png', dpi=220)
        fig.savefig(output_dir / 'CKA_RESULTS.svg', metadata={'Date': None})
        plt.close(fig)
    else:
        # Do not leave a chart from older results beside an empty current table.
        for name in ('CKA_RESULTS.png', 'CKA_RESULTS.svg'):
            (output_dir / name).unlink(missing_ok=True)
    return completed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=ROOT / 'experiments/cka/outputs')
    p.add_argument('--seed', type=int, help='Require this seed for every formal result.')
    a = p.parse_args()
    summarize(a.output_dir, a.seed)


if __name__ == '__main__':
    main()
