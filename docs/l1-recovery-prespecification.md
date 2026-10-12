# Prespecification — L1 artifact recovery (NOT YET RUN)

**Written 2026-10-12, before any run.** Git history is the evidence. Nothing in
this file has been executed.

## What this is, and what it is not

The crop-randomised (L1) campaign saved per-run metrics but **no prediction dumps
and no checkpoints** (ledger §25). Its protocol gap is therefore frame-pooled only,
carries no interval, and cannot be recomputed at video level.

This is **artifact recovery**, not a new analysis. The point estimate already
exists. We are re-collecting the per-crop scores that should have been kept, so the
result can be reported at the same standard as the other four research questions.

❌ It is **not** an attempt to revise the estimate.
❌ It does **not** remove the confound. L1 against L2 still mixes prior exposure
with a different test partition. **V2 remains the headline for that question.**

## Scope

| | |
|---|---|
| Arm to regenerate | crop-randomised only (`results/in_domain_c40/`, dataset `ffpp_c40`) |
| Not touched | the video-disjoint side already has 22 committed dumps |
| Minimal run | Xception, c40, seeds 0–4 — about **2 GPU-hours** |
| Fuller run | the original 9-run configuration — about **3–4 GPU-hours** |

## Expected result, recorded in advance

| quantity | value |
|---|---|
| Original Xception protocol gap | **+0.1764** |
| Range across three architectures | **17.6 – 19.2 points** |

Writing this down first is what makes a disagreement interpretable instead of
negotiable.

## 🚨 Pin the training code to the original commit

Every committed dump in this repo is named `ffpp_*`. By `docs/REPRODUCIBILITY.md`,
all `ffpp_*` runs predate commit `7bb4e11` and use a **single undifferentiated
weight-decay parameter group**. Only the two `ffppfix_*` diagnostic runs use the
corrected grouping.

So the old grouping is not a quirk of the L1 campaign. **It is what the entire
programme ran on, and that consistency is what makes cross-campaign comparison
legitimate.**

Running the recovery against current code would introduce the one run that differs
from all 22 dumps. A discrepancy against +0.1764 would then have three candidate
causes — session drift, the weight-decay change, or ordinary run-to-run variance —
and no way to separate them.

✅ Pin to the original commit and add only the dump step.
❌ Do not run against current `main` and "note the difference".

## Interpretation rule — fixed before running

| outcome | what we report |
|---|---|
| New gap within ordinary run-to-run variance of +0.1764 (reference: largest observed movement is **0.0540**, C2) | Report the recovered value with its interval. The original stands, now better measured |
| New gap outside that range | Report **both**, and treat the discrepancy as a **finding about cross-session reproducibility**, not as a correction. This is on-thesis, not an embarrassment |
| Runs fail to complete | Report nothing new. The original frame-pooled figure stands with its stated limitation |

❌ Under no outcome is the original number quietly replaced.

## Safety conditions

1. **Write to `results/predictions_l1/`.** Never into `predictions_v8/` or
   `predictions_v2/`. The night-1 driver already had a bug where two prediction
   calls shared an output path and one silently overwrote the other
   (`docs/b2-rescore.md`). This is a demonstrated failure mode here, not a
   hypothetical one.
2. **Run `python -m src.reproduce` before and after.** It must return **16/16**
   both times. That is proof nothing moved, rather than assurance.
3. **Distinct `--out-dir` per run**, and check for 5 (or 9) distinct md5 hashes
   before trusting the output.

## What stays untouched

| research question | source | affected? |
|---|---|---|
| Q2 prior exposure, +0.1998 [+0.1422, +0.2542] | `results/predictions_v2/` | **no** |
| Q3 aggregation | `results/predictions_v8/` | **no** |
| Q4 training variation | `results/predictions_v8/` | **no** |
| Q5 verdict, −0.0092 [−0.0358, +0.0180] | `canonical.json` ← `predictions_v8` | **no** |

V2 derives from the **L2** seed-0 manifest via `src/seen_unseen.py`. The
crop-randomised campaign is a separate branch and nothing in that chain reads from
it.

## Why this was not done sooner, and why it is defensible now

Ledger §25 decided against re-running L1, on the grounds that V2 measures the same
phenomenon under control and its lower bound is 10.2× the reference effect. That
reasoning still holds for the **scientific** question.

What changed is the framing. The remaining argument for recovery is narrower and
honest: the protocol gap is currently reported using the **frame-pooled estimand
this paper criticises**, and two GPU-hours removes that inconsistency. That is a
presentational defect in a superseded result, which is why this is optional and
why V2 keeps the headline either way.
