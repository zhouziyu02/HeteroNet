# Generation

Run the commands below from the repository root. The generation loader requires
NumPy, SciPy, scikit-learn and PyTorch. Use a CUDA-compatible PyTorch installation
for training. The supplied trajectory file can be loaded without installing a
physics simulator.

## MuJoCo: supplied data and default configuration

`irregular_generation_diffmn/table1_data/mujoco_training_36.pt` contains 4,620 real
HopperPhysics trajectories with shape `[4620, 36, 14]`. The companion provenance
file records the data seed, simulator versions and SHA256. These trajectories
were regenerated with the TimeCraft Diff-MN physics recipe; they are not claimed
to be identical to another benchmark's original data file.

The default benchmark entry point uses the supplied length 36 and nominal
missing ratio 0.5:

```bash
GPU=0 SEED=1 bash Generation/scripts/mujoco.sh
```

This script trains the conditioned diffusion model and computes generation
benchmark metrics. To reproduce the Generation CKA training and representation
extraction protocol, use its dedicated adapter instead:

```bash
python -m experiments.cka.generation_task --device cuda --seed 1
```

Both use the reported architecture and noise MSE plus `0.05 * x0 MSE` loss.
The CKA adapter omits the benchmark script's preliminary baseline evaluation;
the two entry points are not claimed to produce identical random-number streams.

Lengths 12 and 24 require explicit data preparation. Install the official
Hopper-only physics dependencies, then generate the missing files:

```bash
python -m experiments.cka.install_physics
python -m experiments.cka.prepare_mujoco --lengths 12 24
SEQ_LENS="12 24" MISSING_RATIOS="0.3 0.5 0.7" GPU=0 bash Generation/scripts/mujoco.sh
```

The installer uses official dm-control 1.0.46 and MuJoCo 3.13.0. It omits the
unused `labmaze` dependency, which has no Python 3.13 wheel, and verifies the real
Hopper environment. The generator preserves existing trajectory files. Missing
files cause an error before training; no synthetic MuJoCo proxy is available.
`DATA_ROOT=/absolute/path/to/data` can select another dataset directory for the
shell scripts; direct Python commands use `--data_root` for the benchmark
trainer or `--data-root` for the CKA adapter.

## Stocks and Energy: external real CSVs

These datasets are not bundled. Supply the corresponding Diff-MN numeric CSV,
with one header row and enough observations for the requested window lengths:

```text
Generation/irregular_generation_diffmn/table1_data/stock_data.csv
Generation/irregular_generation_diffmn/table1_data/energy_data.csv
```

```bash
GPU=0 bash Generation/scripts/stocks.sh
GPU=0 bash Generation/scripts/energy.sh
```

Each script retains the original grid of lengths 12/24/36 and missing ratios
0.3/0.5/0.7. Missing, nonnumeric, nonfinite or too-short input fails explicitly;
the loader does not generate substitute Stocks or Energy data.

## Sines

Sines is synthetic by definition. Its original seeded sinusoidal construction
is unchanged and requires no external files:

```bash
GPU=0 bash Generation/scripts/sines.sh
```

Data-method reference: [TimeCraft Diff-MN HopperPhysics](https://github.com/microsoft/TimeCraft/blob/main/Diff-MN/datasets/mujoco_physics.py).
