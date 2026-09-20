# Third-party notices

The following references and copyright notices concern upstream work, not the
authors of this submission. Original notices are retained in the corresponding
source files.

- **EasyTPP**: the vendored `Temporal Point Process/EasyTemporalPointProcess/`
  subset is adapted from [EasyTPP](https://github.com/ant-research/EasyTemporalPointProcess).
  Its [Apache-2.0 license](Temporal%20Point%20Process/EasyTemporalPointProcess/LICENCE)
  and [NOTICE](Temporal%20Point%20Process/EasyTemporalPointProcess/NOTICE) are retained.
  This release removes unused models/search tools and adds a focused Taxi entry point.
- **Latent ODE / time-series data utilities**: several files in `lib/` retain
  their upstream attribution to Yulia Rubanova and Ricky Chen. See
  [Latent ODE](https://github.com/YuliaRubanova/latent_ode),
  [time-series-datasets](https://github.com/rtqichen/time-series-datasets), and the
  retained [MIT license](licenses/latent-ode-MIT.txt).
- **TimeCraft / Diff-MN**: the physical data regeneration recipe follows
  [HopperPhysics](https://github.com/microsoft/TimeCraft/blob/main/Diff-MN/datasets/mujoco_physics.py).
  The upstream [Microsoft MIT license](licenses/timecraft-MIT.txt) is retained.
- **Activity data**: Vidulin et al., Localization Data for Person Activity,
  [DOI 10.24432/C57G8X](https://doi.org/10.24432/C57G8X),
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  The bundled file is a processed form of that dataset; see [data notes](docs/DATA.md).
- **CKA method**: [Kornblith et al., ICML 2019](https://proceedings.mlr.press/v97/kornblith19a.html).
  The numerical implementation is independently checked against the centered
  Gram-matrix formula.

Upstream licenses apply to their respective components; no repository-wide
license is inferred from these third-party notices.
