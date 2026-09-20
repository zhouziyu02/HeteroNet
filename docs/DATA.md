# Data

Included: Activity (`data/activity/processed/data.pt`), Taxi
(`Temporal Point Process/taxi/{train,dev,test}.pkl`), and MuJoCo
(`Generation/irregular_generation_diffmn/table1_data/mujoco_training_36.pt`).
Checksums: [DATA_SHA256.json](DATA_SHA256.json).

MuJoCo contains 4,620 real Hopper trajectories of shape 36×14, regenerated with
seed 123 using dm-control 1.0.46 / MuJoCo 3.13.0. Source and versions are recorded
in the adjacent provenance JSON. To prepare other sequence lengths:

```bash
bash Generation/scripts/prepare_mujoco.sh 12 24 36
SEQ_LENS="12 24 36" MISSING_RATIOS="0.3 0.5 0.7" bash Generation/scripts/mujoco.sh
```

For classification, supply the processed records, outcomes and fixed splits:

```text
data/P12data/processed_data/PTdict_list.npy
data/P12data/processed_data/arr_outcomes.npy
data/P12data/splits/phy12_split1.npy
data/P19data/processed_data/PT_dict_list_6.npy
data/P19data/processed_data/arr_outcomes_6.npy
data/P19data/splits/phy19_split1_new.npy
data/PAMAP2data/processed_data/PTdict_list.npy
data/PAMAP2data/processed_data/arr_outcomes.npy
data/PAMAP2data/splits/PAMAP2_split_1.npy
```

Other optional data paths:

```text
data/physionet/processed/set-{a,b,c}_0.0.pt
data/ushcn/processed/ushcn.pt
data/mimic/mimic.pt
Generation/irregular_generation_diffmn/table1_data/stock_data.csv
Generation/irregular_generation_diffmn/table1_data/energy_data.csv
```

Data formats are defined in `lib/`. Stock/Energy files must be numeric CSVs with
a header. MIMIC is access-controlled and is not distributed. See
[third-party sources and licenses](../THIRD_PARTY_NOTICES.md).
