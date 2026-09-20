# HeteroNet — Reproducibility Supplement

Core implementation and task entry points for irregular time-series classification,
interpolation, extrapolation, temporal point processes, and generation. The model
class retains the implementation name `ITSPM`.

Three datasets are included: **Activity, Taxi, and MuJoCo** (about 37 MiB total).
No pretrained checkpoints are required. Paired representations from the four CKA
experiments are included for immediate numerical verification.

## Setup

Use Python 3.11 or later and a matched PyTorch/torchvision installation. Full
training requires one CUDA GPU; the included CKA artifacts can be verified on CPU.
The reference CKA run used a Tesla T4, Python 3.13, and PyTorch 2.11.0+cu128.

```bash
python -m pip install -r requirements.txt
python -m unittest experiments.cka.test_common -v
```

On Colab, enable a GPU runtime and retain its preinstalled PyTorch/torchvision pair.
MuJoCo simulation packages are only needed when regenerating physical trajectories;
training on the included tensor does not require them.

## Reproduce the four CKA experiments

Run from the repository root:

```bash
python -m experiments.cka.run_suite --seed 1
```

This trains and evaluates Interpolation/Activity, Extrapolation/Activity, TPP/Taxi,
and Generation/MuJoCo sequentially. Outputs are written to `experiments/cka/outputs/`.
Interrupted runs resume from saved training checkpoints. Changed configurations or
source files require a new `--output-dir`.

To recompute the supplied CKA values **without training or a GPU**:

```bash
python -m experiments.cka.summarize --output-dir results/cka --seed 1
```

| Task | Dataset | Centered linear CKA | Paired test samples |
|---|---|---:|---:|
| Interpolation | Activity | 0.022326 | 1,075 |
| Extrapolation | Activity | 0.034712 | 1,075 |
| TPP | Taxi | 0.257396 | 2,048 |
| Generation | MuJoCo | 0.485164 | 924 |

These compare the masked mean of early event embeddings with the final backbone
representation used by each task. They are **not** first-versus-last Transformer
block comparisons. See [CKA protocol](docs/CKA.md) for endpoints, model selection,
and interpretation. All values are from one training seed, not multi-seed averages.

## Task-specific training

The scripts below keep the original benchmark configurations. Their seeds differ
from the unified seed=1 CKA experiment above. `GPU` and `SEED` can be overridden.

```bash
# Included Activity data
GPU=0 bash Interpolation/scripts/activity.sh
GPU=0 bash Extrapolation/scripts/activity.sh

# Included Taxi data; validation selects the checkpoint
python "Temporal Point Process/EasyTemporalPointProcess/examples/train_nhp.py" --device cuda --seed 1 --output-dir outputs/tpp_taxi

# Included physical MuJoCo data: length 36, missing rate 0.5
GPU=0 SEED=1 bash Generation/scripts/mujoco.sh

# Synthetic sines need no external data
GPU=0 SEED=1 bash Generation/scripts/sines.sh
```

Classification and the other benchmark datasets require external processed data.
See [data layout and preparation](docs/DATA.md) before using their scripts:

```bash
GPU=0 bash Classification/scripts/P12.sh
GPU=0 bash Classification/scripts/P19.sh
GPU=0 bash Classification/scripts/PAM.sh
```

## Repository layout

```text
models/                         shared ITSPM backbone
lib/                            data loaders and task utilities
classification.py, regression.py
Classification/, Interpolation/, Extrapolation/  task scripts
Temporal Point Process/         minimal EasyTPP integration and Taxi data
Generation/                     diffusion implementation, scripts and MuJoCo data
experiments/cka/                 training, capture, CKA and verification
results/cka/                    reference arrays, metrics and training traces
docs/                           data requirements and experimental protocol
```

The numerical reference arrays are included; checkpoints, historical search runs,
machine logs, and local account configuration are excluded. Original data hashes
are listed in `docs/DATA_SHA256.json`.

## Third-party material

Third-party code and data retain their original attribution and licenses; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Those notices identify upstream
projects, not the authors of this submission.
