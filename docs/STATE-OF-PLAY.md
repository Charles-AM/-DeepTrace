# State of play — 2026-09-06

One page to resume from after a break. Read `ESSENCE.md` for what the paper
claims; this is where the work actually stands.

## Where we are

**Research ~90%. Manuscript 0%.** No `.tex`, no publication figures. The
experimental phase is nearly closed; what remains is mostly writing.

81 training runs, 66,000 committed predictions, ~45 GPU-hours.

## The four rungs — all measured

| # | claim | evidence |
|---|---|---|
| 1 | Protocol changes apparent performance (~18 pts; gap grows 9.5–10.4 pts under c40) | `results/in_domain_c40_vid/`, `_c23_vid/` |
| 2 | Resampling unit changes the conclusion without changing the estimate | `results/analysis/cluster_boot/`, `resolution/` |
| 3 | Complete pipeline runs move the estimated architectural effect | `results/in_domain_c40_fixedsplit/` |
| 4 | Compression associated with wider intervals on identical test content | `results/analysis/resolution/` §2 |

Primary number: **component half-widths 0.037–0.062 AUC = 2.6–4.5× the +0.014
reference effect**, across five pipeline runs.

## Blocked on GPU quota

In priority order — `docs/post-v8-queue.md` has the cells.

| # | item | cost | why |
|---|---|---|---|
| 1 | **V2** seen vs unseen, matched | ~80 min | turns "protocol gap" into a leakage measurement; code built and tested (`src/seen_unseen.py`, 8 tests) |
| 2 | **G7a** late fusion at c23 on L2 checkpoints | ~15 min | **main-paper** figure currently computed on leaky checkpoints |
| 3 | **Repeat audit** 2× seed-0 in one session | ~146 min | separates run-level from session-level; code built and dry-run (`docs/repeat_audit.py`) |
| 4 | **G7b** CKA on L2 checkpoints | ~10 min | supplementary, same defect |
| 5 | **G3** `band_ablation.py` smoke test | ~5 min | last register gap |

## Blocked on nothing (CPU, can be done any time)

- **Crossed seed × component interval — the highest-value item left, and CPU-only.**
  Before V8 every seed drew a different split, so a crossed bootstrap was
  impossible: the same component sample cannot be applied across seeds that do not
  share a test set. V8's five models share one verified split, so it is computable
  for the first time. Blocked *only* on pulling the V8 `preds/` out of the Kaggle
  version output and committing them, as was done for V1. The V8 numbers in the
  repo are frame-pooled only.
- **Independent-samples cluster bootstrap** for V2 — one model on two different
  test sets is not the paired case `cluster_boot` handles. To write when V2 lands.

`src/crossed_boot.py` is **written and tested** (7 tests, synthetic data). It runs
the moment the V8 dumps are in `results/predictions/`:

```
python -m src.crossed_boot \
  --a-glob 'results/predictions/ffpp_c40_vid_xception_seed*_test.csv' \
  --b-glob 'results/predictions/ffpp_c40_vid_xception_fad_seed*_test.csv' \
  --margins 0.014 --out-dir results/analysis/crossed
```

It refuses to run if the runs were not scored on identical test items — the
condition V8's fixed split exists to provide — and reports the crossed interval
beside the two conditional ones (components only, seeds only) so the cost of
crossing is visible.
- **Publication figures**: resolution curve, protocol gap, three-unit comparison.
- **The manuscript.**

## Traps to remember

- **No `--amp`.** `run_ablation.py` never passed it, so every comparison run is
  fp32. It cost a 155-minute run once. Commands now build from
  `results/reference/train_args_c40_vid.json` (`src/reference_cmd.py`, 7 tests).
- **Kaggle outputs vanish.** The V1 dumps were lost and recovered only from a
  `~/Downloads` tarball. Commit prediction CSVs to the repo every time.
- **Save & Run All runs in its own container** — its `/kaggle/working` is not
  visible from an interactive session. Results come from the version output.
- **DO NOT CITE**: `results/analysis/late_fusion/`, `cka/`, `permanip/` (L1
  checkpoints), and `results/in_domain_c40_fixedsplit/` history commit `c131486`
  (AMP). `spectra/` is fine — it loads no checkpoint.
- **pytest and sklearn are not installed locally.** Tests run through a shim in
  the session scratchpad; the real suite runs on Kaggle. `cluster_boot` now uses
  the internal rank AUC, verified bit-identical to sklearn on all 8 V1 outputs.

## Wording that is settled

Thesis: ESSENCE §0, revised 2026-09-06. Use **"fixed-split training-run
variability"**, never "optimisation variability" or "bound". The repeat finding is
an **audit (n=1)** with no causal attribution. Until V2 lands it is a **protocol
gap**, not leakage.

## Gap register

Closed: G1, G2, G4, G6, permanip half of G7.
Open: G3 (`band_ablation` smoke test), G5 (V8 point estimate only — not
distinguishable at n=5), G7 (late fusion + CKA).
