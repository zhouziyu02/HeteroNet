# HeteroNet

Python 3.11+ and a CUDA GPU. Activity, Taxi and MuJoCo are included.
Prepare other datasets as described in [docs/DATA.md](docs/DATA.md), then run:

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

Use `GPU=0 SEED=1 bash ...` to override defaults. [Third-party notices](THIRD_PARTY_NOTICES.md).
