# Generation

The generation entry point implements the revised **latent diffusion head**.
It uses `models/HeteroNet.py` as its encoder and trains in two stages:

1. Train the HeteroNet encoder and time-query decoder with reconstruction MSE
   at observed training entries only.
2. Freeze both modules, standardize the encoded training latents, and train a
   latent DDPM with noise-prediction MSE.

Sampling starts from Gaussian noise. The generated latent is transformed back
using the saved training-latent mean and standard deviation, then decoded onto
the regular output grid. The sampler has no evaluation-observation or mask
input. Generated sequences are not clamped to reference values or calibrated
to reference marginal distributions.

## Entry points and settings

Run the existing scripts from the repository root:

```bash
bash Generation/scripts/sines.sh
bash Generation/scripts/stocks.sh
bash Generation/scripts/energy.sh
bash Generation/scripts/mujoco.sh
```

They now select `heteronet_latent_diffusion`. Their dataset choices, window
lengths, missing rates, seeds, sample counts, batch sizes, diffusion steps,
training epochs, and model widths retain their existing values. The AE stage
uses the existing `--ae_epochs` and `--ae_lr` defaults unless supplied explicitly.
The old conditional/direct generation branches and their reconstruction,
clamping, and calibration switches are no longer part of this entry point.

The data loaders and their benchmark min-max preprocessing are unchanged.
The additional z-score transform is fitted only on observed entries of the
training partition. Complete evaluation references are retained for scoring,
but are not encoded or passed to either generator training stage.

The existing metric implementations and their settings are retained. The
Diff-MN constants in the script are quoted published reference values, not
results of running the revised head. Existing conditional-head checkpoints
and scores should not be relabelled as latent-head results.

## Code map

All definitions below are in
[`irregular_generation_diffmn/train_irregular_generation.py`](irregular_generation_diffmn/train_irregular_generation.py).

| Paper operation | Implementation |
| --- | --- |
| Observed-entry reconstruction | `observed_reconstruction_loss` |
| Freeze encoder and decoder | `freeze_autoencoder` |
| Training-latent standardization | `standardize_training_latents` |
| Latent noise prediction | `LatentDenoiser` and `LatentDDPM.q_sample` |
| Noise-only generation and inverse latent transform | `sample_sequences` |
| Save/load all sampling parameters and statistics | `save_generation_checkpoint`, `sample_from_checkpoint` |

The checkpoint contains both networks, their architecture settings, the
diffusion settings, and the data/latent normalization statistics. It can be
used without loading a reference dataset:

```python
from pathlib import Path
import torch
from Generation.irregular_generation_diffmn.train_irregular_generation import sample_from_checkpoint

generated = sample_from_checkpoint(
    Path("path/to/model.pt"), n_samples=100,
    device=torch.device("cpu"), seed=1,
)
```

## Lightweight checks

```bash
python3 -m unittest discover -s Generation/tests -v
```

These checks exercise loss masking, training-only statistics, frozen modules,
sampling inputs, checkpoint round-tripping, and CLI defaults. They do not run
training epochs, dataset benchmarks, or discriminator evaluation.
