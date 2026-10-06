# HeteroNet

The task scripts default to a CUDA GPU. Python 3.11 is verified; CPU execution
is also available for interface checks and small runs. Install a matching
PyTorch/torchvision pair for your device, then install dependencies, prepare
the local data listed below, and run from the repository root:

```bash
pip install -r requirements.txt
bash Classification/scripts/P12.sh        # classification: external data
bash Classification/scripts/P19.sh
bash Classification/scripts/PAM.sh
bash Interpolation/scripts/activity.sh    # interpolation
bash Extrapolation/scripts/activity.sh    # extrapolation
bash "Temporal Point Process/scripts/taxi.sh"  # temporal point process
bash Generation/scripts/mujoco.sh         # generation
bash Generation/scripts/sines.sh
bash Generation/scripts/stocks.sh         # external CSV
bash Generation/scripts/energy.sh         # external CSV
```

Use `GPU=0 SEED=1 bash ...` to override a script's defaults. Architecture,
training budgets, and checkpoint selection are defined by the task scripts
and their entry points. The TPP entry point currently covers Taxi only.
Classification and regression normalization statistics use the training
split. Checkpoints are selected by validation metrics. Regression saves the
best checkpoint under `save/` by default; use `--test_only --load_path PATH`
with the same dataset, task, and model configuration to evaluate it.
Regression reports and excludes windows without observed inputs or prediction
targets after assigning records to their data splits.

Generation trains an encoder/decoder on observed entries, freezes them, and
fits a DDPM to standardized training latents. Sampling uses noise and saved
normalization statistics, without evaluation observations or marginal
calibration. Published comparison constants are reference values, not runs
of this implementation; checkpoints and scores from a different head are
not interchangeable.
The reported `gaussian_baseline` fits all training values, including values
hidden by the artificial observation mask. It is a complete-data reference
baseline and does not follow the generator's observed-only input protocol.

## Local data

Included files and SHA-256 checksums are listed in `DATA_MANIFEST.json`:
Activity (`data/activity/processed/data.pt`), Taxi
(`Temporal Point Process/taxi/{train,dev,test}.pkl`), and MuJoCo
(`Generation/irregular_generation_diffmn/table1_data/mujoco_training_36.pt`).
MuJoCo is regenerated HopperPhysics data, not a bit-identical copy of the
original benchmark release. Its adjacent provenance file records the seed,
simulator versions, and checksum. Other lengths can be prepared with:

```bash
bash Generation/scripts/prepare_mujoco.sh 12 24 36
SEQ_LENS="12 24 36" bash Generation/scripts/mujoco.sh
```

Supply the following files for the remaining tasks:

```text
data/P12data/processed_data/{PTdict_list,arr_outcomes}.npy
data/P12data/splits/phy12_split1.npy
data/P19data/processed_data/{PT_dict_list_6,arr_outcomes_6}.npy
data/P19data/splits/phy19_split1_new.npy
data/PAMAP2data/processed_data/{PTdict_list,arr_outcomes}.npy
data/PAMAP2data/splits/PAMAP2_split_1.npy
data/physionet/processed/set-{a,b,c}_0.0.pt
data/ushcn/processed/ushcn.pt
data/mimic/mimic.pt
Generation/irregular_generation_diffmn/table1_data/{stock_data,energy_data}.csv
```

Formats are defined in `lib/`; Stock/Energy CSVs require numeric columns and
a header. MIMIC is access-controlled and is not bundled. Data loaders use
local files only. To preprocess raw Activity or PhysioNet data, place
`ConfLongDemo_JSI.txt` in `data/activity/raw/`, or `set-{a,b,c}.tar.gz` in
`data/physionet/raw/`, respectively. Missing files produce an explicit error.

Third-party attribution and license locations are in `THIRD_PARTY_NOTICES.md`.

## Small checks

Run the CPU regression checks from the repository root:

```bash
python3 -m unittest discover -s tests -v
python3 -m unittest discover -s Generation/tests -v
python3 -m unittest discover -s "Temporal Point Process/EasyTemporalPointProcess/tests" -v
```

These cover input shapes, masks, metrics, checkpoint loading, and small
training/sampling runs. They do not reproduce full training budgets or verify
published benchmark scores. Corrections to normalization, metrics, and TPP
pooling can change results relative to earlier code versions. The checks were
run with Python 3.11,
PyTorch 2.3.1, NumPy 1.26.1, and scikit-learn 1.5.2.
