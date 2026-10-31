# NPR campaign — local rehearsal before the Kaggle session

**Date:** 2026-10-31. **Status:** rehearsal only. No campaign result exists.

⚠️ **Nothing here is a finding about NPR.** The rehearsal ran on 360 synthetic
noise crops for one epoch without pretrained weights, purely to prove the
pipeline executes. Its AUCs are meaningless and must never be reported.

The purpose was to find the failures that would otherwise be found by burning a
Kaggle session. It found two.

## What was rehearsed

Full chain on CPU, both arms, one seed:

| step | result |
|---|---|
| `src.train --config xception` | completed, checkpoint written |
| `src.train --config xception_npr` | completed, checkpoint written |
| `src.predict` both arms | 36 predictions each, `config` stamped correctly |
| `src.campaign_gate` | ✅ passed, 11/11 checks |

## 🚨 Finding 1 — `XceptionNPR` was missing `get_alpha`

`src/train.py` calls `model.get_alpha()` unconditionally after every epoch.
`XceptionNPR` did not implement it.

The model built, reported the expected 20,811,050 parameters, passed all eight
transform-parity tests, and **trained a complete epoch** before crashing with
`AttributeError` at the first logging step. On Kaggle that is one epoch of GPU
time spent to reach a crash, on the first run of the campaign.

The existing tests could not catch it: they exercised `npr_residual` directly and
never built through `build_model`, so nothing asserted the interface the shared
training loop depends on. Fixed in `035dbf9`, with `xception_npr` added to the
parametrized interface contract test so the gap cannot reopen.

## 🚨 Finding 2 — `requirements.txt` could not build a working environment

`pillow` was undeclared although `src/data.py` loads every crop through PIL, and
`tensorboard` was undeclared although `src/train.py:19` imports
`torch.utils.tensorboard` at module scope. **Training died at import.**

`tensorboard` is the instructive case: its top-level module is `torch`, which *is*
declared, so no scan of top-level import names can see the gap. That is why it
survived this long. Fixed in `aa67e18` with a test that asserts it specifically.

Since the paper claims a reproducibility spine, a requirements file that cannot
build a runnable environment is a defect in the contribution, not packaging trivia.

## Not a finding: MPS `nan` metrics

On Apple Silicon (`device=mps`) the runs reported `val_auc=nan` or raised
`ValueError: Target is multiclass but average='binary'` from scikit-learn. Both
symptoms are inconsistent with each other and both disappeared on `device=cpu`,
where the same data and code gave `val_auc=1.0000` for the baseline arm.

The label tensor is clean where it is produced — the loader yields `int64` with
`unique=[0 1]` and 18/18 balance — so the corruption happens after transfer,
consistent with MPS's incomplete int64 support. **Kaggle runs CUDA, so this does
not affect the campaign.** Recorded only so the symptom is not re-investigated.

❌ Do not read this as a claim about CUDA or about the repository's metrics code.
It was not tested on CUDA here, and nothing was changed in `src/metrics.py`.

## What the rehearsal did NOT verify

- ⚠️ **Per-epoch cost.** The prespecified timing gate is unrun. CPU timings
  (146 s and 173 s per epoch on 288 synthetic crops) say nothing about a T4 on the
  real c40 set. `docs/npr_run.py --time-one-epoch` remains required before the
  campaign, and the 4–6 GPU-hour figure remains an extrapolation.
- Real data. No FF++ crop was touched.
- Convergence, accuracy, or any comparison between the arms.
- The 10-run counterbalanced loop end to end; only its command construction was
  checked, via `--dry-run`.

## Protected files

`git diff` on `src/train.py`, `src/crossed_boot.py`, `src/agg_space.py`,
`src/cluster_boot.py` and `src/clusters.py` is empty. The rehearsal invoked them;
it did not change them.
