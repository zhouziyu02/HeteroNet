# CKA: early event and final backbone representations

Centered linear CKA (Kornblith et al., ICML 2019), globally centered over paired held-out samples.

| Task | Dataset | Linear CKA | Paired samples | Seed | Status |
|---|---|---:|---:|---:|---|
| Interpolation | Activity | 0.022326 | 1075 | 1 | completed |
| Extrapolation | Activity | 0.034712 | 1075 | 1 | completed |
| TPP | Taxi | 0.257396 | 2048 | 1 | completed |
| Generation | MuJoCo | 0.485164 | 924 | 1 | completed |

First endpoint: EventEncoder output, masked mean over observed events per sample. Last endpoint: the final backbone representation consumed by the task head. See each results.json for exact endpoints.

Rows are held-out windows (Activity), next-event contexts (Taxi), or trajectories (MuJoCo). These are descriptive similarities from one training seed, not predictive accuracy or a mean across seeds.
