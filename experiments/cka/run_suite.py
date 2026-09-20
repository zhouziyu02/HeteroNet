"""Run the four CKA experiments sequentially on one CUDA GPU."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]

TASK_NAMES = {'interpolation': ('interpolation_activity', 'activity'),
              'extrapolation': ('extrapolation_activity', 'activity'),
              'tpp': ('tpp_taxi', 'taxi'), 'generation': ('generation_mujoco', 'mujoco')}


def write_json(path, payload):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def request_manifest(task, seed, max_samples):
    """Include adapter/default/model code so changed configs cannot reuse output."""
    adapter = 'regression_task.py' if task in ('interpolation', 'extrapolation') else f'{task}_task.py'
    paths = [ROOT / 'experiments/cka' / name for name in (adapter, 'common.py', 'requirements-colab.txt')]
    paths.append(ROOT / 'requirements.txt')
    if task == 'tpp':
        paths.extend((ROOT / 'Temporal Point Process/EasyTemporalPointProcess').rglob('*.py'))
        paths.append(ROOT / 'Temporal Point Process/EasyTemporalPointProcess/examples/configs/itspm_tpp_config.yaml')
    else:
        paths.extend((ROOT / 'models').rglob('*.py'))
        if task == 'generation':
            paths.extend([ROOT / 'Generation/irregular_generation_diffmn/train_irregular_generation.py',
                          ROOT / 'experiments/cka/prepare_mujoco.py'])
        else:
            paths.extend((ROOT / 'lib').rglob('*.py'))
    source_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted(set(paths))}
    return {'schema': 1, 'task': task, 'dataset': TASK_NAMES[task][1], 'seed': seed,
            'max_samples': max_samples, 'device': 'cuda', 'source_sha256': source_hashes}


def check_manifest(output_dir, manifest):
    path = output_dir / 'suite_request.json'
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise RuntimeError(f'{output_dir} belongs to different seed/sample/config/source settings; use a new output directory.')
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise RuntimeError(f'{output_dir} contains an unverified prior run without suite_request.json; use a new output directory.')
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(path, manifest)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--max-samples', type=int, default=2048)
    p.add_argument('--output-dir', type=Path, default=ROOT / 'experiments/cka/outputs')
    p.add_argument('--tasks', nargs='+', choices=['interpolation', 'extrapolation', 'tpp', 'generation'],
                   default=['interpolation', 'extrapolation', 'tpp', 'generation'])
    a = p.parse_args()
    if a.max_samples < 2:
        p.error('--max-samples must be at least 2 for all four task adapters')
    if len(set(a.tasks)) != len(a.tasks):
        p.error('--tasks must not contain duplicates')
    a.output_dir = a.output_dir.resolve()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    status = {'state': 'running', 'started_at_unix': time.time(), 'seed': a.seed,
              'tasks': {}, 'requested_tasks': a.tasks, 'max_cka_samples': a.max_samples}
    write_json(a.output_dir / 'suite_status.json', status)
    shared = ['--seed', str(a.seed), '--max-samples', str(a.max_samples), '--device', 'cuda']
    try:
        import torch
        from experiments.cka.summarize import validate_result, summarize
        if not torch.cuda.is_available():
            raise RuntimeError('Full CKA training requires a CUDA GPU. Use the documented CPU commands for artifact verification.')
        status.update(gpu=torch.cuda.get_device_name(0), torch=str(torch.__version__), cuda=torch.version.cuda)
        packages = sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions() if d.metadata['Name'])
        (a.output_dir / 'environment.txt').write_text('\n'.join(f'{n}=={v}' for n, v in packages) + '\n')
        for task in a.tasks:
            name, dataset = TASK_NAMES[task]
            out = a.output_dir / name
            status['current_task'] = task
            status['tasks'][task] = {'state': 'running', 'started_at_unix': time.time()}
            write_json(a.output_dir / 'suite_status.json', status)
            check_manifest(out, request_manifest(task, a.seed, a.max_samples))
            if (out / 'results.json').exists():
                report, _ = validate_result(out / 'results.json', task, dataset, a.seed)
                status['tasks'][task].update(state='completed', reused=True, result=str(out / 'results.json'),
                                             linear_cka=report['linear_cka'], n_samples=report['n_samples'])
                write_json(a.output_dir / 'suite_status.json', status)
                summarize(a.output_dir, a.seed)
                continue
            if task in ('interpolation', 'extrapolation'):
                cmd = [sys.executable, '-u', '-m', 'experiments.cka.regression_task', '--task', task,
                       '--output-dir', str(out), *shared]
                if (out / 'latest.pt').exists():
                    cmd.append('--resume')
            elif task == 'tpp':
                cmd = [sys.executable, '-u', '-m', 'experiments.cka.tpp_task', '--output-dir', str(out), *shared]
                if (out / 'latest.pt').exists():
                    cmd.append('--resume')
            else:
                if not (ROOT / 'Generation/irregular_generation_diffmn/table1_data/mujoco_training_36.pt').exists():
                    subprocess.run([sys.executable, '-u', '-m', 'experiments.cka.install_physics'], cwd=ROOT, check=True)
                subprocess.run([sys.executable, '-u', '-m', 'experiments.cka.prepare_mujoco', '--lengths', '36'], cwd=ROOT, check=True)
                cmd = [sys.executable, '-u', '-m', 'experiments.cka.generation_task', '--output-dir', str(out), *shared]
            print('START', task, json.dumps(cmd), flush=True)
            status['tasks'][task]['command'] = cmd
            write_json(a.output_dir / 'suite_status.json', status)
            subprocess.run(cmd, cwd=ROOT, check=True)
            report, _ = validate_result(out / 'results.json', task, dataset, a.seed)
            status['tasks'][task].update(state='completed', finished_at_unix=time.time(),
                                         linear_cka=report['linear_cka'], n_samples=report['n_samples'])
            write_json(a.output_dir / 'suite_status.json', status)
            summarize(a.output_dir, a.seed)
        summarize(a.output_dir, a.seed)
        status['state'] = 'completed'
        status['finished_at_unix'] = time.time()
        status.pop('current_task', None)
        write_json(a.output_dir / 'suite_status.json', status)
    except BaseException as exc:
        current = status.get('current_task')
        if current is not None:
            status['tasks'][current].update(state='failed', error=str(exc), finished_at_unix=time.time())
        status.update(state='failed', error=str(exc), finished_at_unix=time.time())
        write_json(a.output_dir / 'suite_status.json', status)
        raise
    print('ALL_REQUESTED_CKA_EXPERIMENTS_COMPLETED', flush=True)


if __name__ == '__main__':
    main()
