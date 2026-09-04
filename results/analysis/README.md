# Analysis experiments (2026-09-03, updated 2026-09-04)

Mechanistic follow-ups on the c23 checkpoints. Sections below marked "no
retraining" reuse existing checkpoints/crops only.

## `late_fusion/seed_results.csv` — C1b, no retraining (external-review addition)

`src/late_fusion.py`: combine `baseline_spatial` and `frequency_only` scores via
logistic regression fit on val, evaluated on test. Sidesteps the gated-fusion
weight-decay confound (see `fusion_alpha.csv` note below) entirely — no gate, no
architecture, just two independently-trained models' scores combined the simplest
possible way.

**Result, 3 seeds:** logreg-fusion AUC is identical to spatial-alone AUC to 4
decimal places on every seed (Δ = 0.0000, −0.0000, −0.0000), despite the fitted
frequency coefficient being nonzero and positive every time. **Checked and closed
(2026-09-04):** re-fit with an effectively unregularised logistic regression
(`C=1e6`) gives the same result (0.9969/0.9927/0.9951 vs 0.9968/0.9927/0.9951) —
default L2 regularisation was not hiding the effect. The naive 50/50 average is
consistently slightly *worse* than spatial alone. **Cleanest available evidence for
"low incremental predictive value"** — no confounds from the gated architecture,
and now no confound from the combiner's regularisation either.

**Provenance:** `seed_results.csv` (both this file and `cka/seed_results.csv`
below) are genuine `src.late_fusion` / `src.cka` output, re-run and re-downloaded
from Kaggle on 2026-09-04 specifically to replace an earlier hand-transcription —
bit-for-bit identical to the transcribed numbers, confirming both scripts are
deterministic and the results reproducible.

## `efficiency/params_flops_latency.csv` — C8, no retraining

Params (M) / GFLOPs / batch-32 GPU latency (ms), matched input size 128px, via
`fvcore`. Two different frequency-branch designs cost very different amounts:

| comparison | params | GFLOPs | latency |
|---|---|---|---|
| `full` vs `baseline_spatial` (ResNet-18 + separate freq branch + gated fusion) | +1.2% | **+31%** | **+44%** |
| `f3net` vs `xception` (FAD = 9-channel input to the *same* backbone, no separate branch) | +0.2% | +2.7% | +3% |

F3-Net's channel-stacking design is nearly free — its accuracy/robustness numbers
alone decide whether it's worth it. Our separate-branch design carries real cost
(44% more latency) on top of zero accuracy/predictive benefit (C1, C1b) — a cleanly
"not worth it" result specific to that architectural choice, not a blanket property
of "adding frequency."

## `cka/seed_results.csv` — C4, representation-level check (2026-09-04)

`src/cka.py`: linear CKA between the spatial-branch and frequency-branch features
(pre-fusion, `full` config) vs a random-projection null baseline, 3 seeds.

**Result: CKA(spatial, frequency) ≈ CKA(spatial, random) on every seed** (0.0051–
0.0077 vs 0.0055–0.0058 — indistinguishable from the no-relationship floor). This
**falsifies** the "the spatial CNN already encodes the same information as
frequency" (redundancy) hypothesis exactly as pre-registered in this file's C4
entry ("falsified if... CKA is low"). Correcting the theory, not hiding the result:

**The two branches learn genuinely distinct (non-redundant) representations — the
frequency branch's information is not duplicated by the spatial branch. It's
simply not discriminative for this task.** `frequency_only` alone gets 0.70 AUC
(some real signal, mostly noise relative to spatial's ~0.997), and that signal is
orthogonal enough to spatial's that combining them (C1, C1b) adds nothing not
because it's redundant, but because it isn't useful. This is a more precise and
more interesting mechanism than "redundant" — replace that word wherever it
appears in the writeup with "non-discriminative" / "adds nothing despite being
representationally distinct."

## `fusion_alpha.csv` — learned gated-fusion weight

`alpha_spatial = sigmoid(alpha_logit)`, the weight on the spatial branch
(`fused = alpha*spatial + (1-alpha)*freq`), init 0.5.

**Result:** alpha = **0.496 ± 0.001** for every gated config (`full`,
`full_banddrop`, `no_mask`) and every seed. The gate never moved from init
(|Δlogit| < 0.023). With the spatial branch already at ~0.997 AUC on c23 the loss
is essentially flat in alpha, so there is no gradient pressure to re-weight. The
"learnable" fusion behaves as a fixed 50/50 average here — corroborates
`baseline_spatial` ≈ `full` and `no_fusion_concat` ≈ `full`.

**Confound check (2026-09-04, `weight_decay_fixed=True` rows):** external review
correctly flagged that `alpha_logit` was getting AdamW weight decay applied
directly to it (no exclusion for scalars/gates — a real bug, fixed in
`utils.no_decay_param_groups`, commit `7bb4e11`), which could have been pulling α
toward 0.5 regardless of the task gradient. Retrained `full` and `no_mask` (seed 0,
`ffppfix_*`) with the fix: α = 0.4947 and 0.4941 — statistically indistinguishable
from the original biased runs, test AUC essentially unchanged (0.9974 vs 0.9969;
0.9972 vs 0.9973). **Weight decay was not the primary driver — the "gate never
engages" finding survives its strongest objection** (1 seed each with the fix; more
would make it airtight, but this is real evidence, not just a caveat removed).

**Provenance (2026-09-04):** `fusion_alpha.csv` above is now byte-for-byte the
genuine output of `src.gate_readout` run on Kaggle (`fusion_alpha_raw.csv` is the
untouched copy) — not a hand transcription of pasted console text. Per-run test
metrics for the two `ffppfix_*` checkpoints are at
`results/in_domain/per_run/ffppfix_{full,no_mask}_seed0.json`, also copied directly
from the generated files (each includes a precise `alpha` field independently
confirming the gate_readout numbers: 0.49474871.../0.49413418...). The actual
`best.pt` checkpoint weights (44MB each) are **not** in git, consistent with every
other checkpoint in this project (see `results/README.md`) — backed up locally at
`~/Downloads/gatefix_artifacts.tar.gz`.

## `permanip/` — per-forgery-method breakdown, 3 seeds

`seed{0,1,2}.csv` long form (run × method: roc_auc/eer/accuracy, real vs that
method only); `auc_seed{0,1,2}.csv` pivots.

**3-seed mean ROC-AUC:**

| model | Deepfakes | Face2Face | FaceSwap | NeuralTextures |
|---|---|---|---|---|
| xception | 0.9991 | **0.9979** | **0.9992** | **0.9944** |
| f3net | 0.9990 | 0.9975 | 0.9989 | 0.9930 |
| baseline_spatial | 0.9985 | 0.9955 | 0.9978 | 0.9877 |
| full | 0.9976 | 0.9953 | 0.9975 | 0.9884 |
| full_banddrop+sas | 0.9978 | 0.9958 | 0.9978 | 0.9882 |
| frequency_only | 0.852 | 0.668 | 0.665 | 0.617 |

- On **NeuralTextures** (the subtlest manipulation — the case frequency methods
  claim to win) the plain spatial Xception is best; the frequency hybrids match the
  spatial baseline, not beat it.
- `frequency_only` does **best on Deepfakes** (crudest, most artifact-heavy) and
  **worst on NeuralTextures** — the opposite of the frequency-detection premise
  that spectral analysis surfaces subtle traces. The spatial CNN already scores
  0.999 on Deepfakes.
- Ordering is stable every seed, every method:
  `xception ≥ f3net > full ≈ full_banddrop+sas ≈ baseline_spatial`.

## `spectra/` — real-vs-fake frequency gap (`src/spectra.py`, 3000 crops/class, native 160 px)

`radial.csv` (azimuthal DFT power + std bands, c23 & jpeg30), `summary.csv`,
`spectra.png` (radial curves + 2-D DCT fake−real difference + per-coefficient
t-map), `dct_maps.npz` (the 2-D arrays).

| metric | c23 | jpeg30 |
|---|---|---|
| radial power gap fake−real, mean | 0.94 dB | 0.79 dB |
| radial gap, high-freq band (norm.f > 0.5) | 1.24 dB | 0.94 dB |
| DCT coefficients with \|t\| > 3 | 5.6 % | 4.8 % |
| DCT coefficients with \|t\| > 5 | 0.04 % | 0.02 % |
| max \|t\| | 5.76 | 5.55 |

- A stationary real-vs-fake spectral gap **does exist** in FF++ c23 (5.6 % of DCT
  coefficients differ at |t|>3 vs ~0.3 % by chance), concentrated at high
  frequency.
- **Effect size is small** — max t ≈ 5.8 at n = 3000/class → Cohen's d ≈ 0.15 at
  the single most discriminative coefficient.
- **JPEG-q30 attenuates it only ~15–25 %** — it does not "destroy" the signal.
- Conclusion: frequency modelling fails here not because the signal is absent or
  compression-fragile, but because it is **marginal and redundant** — a spatial
  CNN already captures whatever discriminative content it holds
  (`baseline_spatial` ≈ `full`; `frequency_only` = 0.70).
- The high-pass **residual** t-map is numerically identical to the raw t-map (the
  Gaussian high-pass only touches the lowest ~8 % of frequencies, where there is no
  signal) — the residual variant adds nothing; can be dropped.

`spectra/` also once held a flawed v1 (`radial_psd.*`): it bilinear-resized crops
to 128 px before the FFT, low-passing away the very signal being measured. Not
kept. `src/spectra.py` was rewritten (commit fda0d73).
