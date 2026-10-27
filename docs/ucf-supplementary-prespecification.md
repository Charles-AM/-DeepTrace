# Prespecification — UCF supplementary run (NOT YET RUN)

**Written 2026-10-27, before any run.** Git history is the evidence. Nothing here
has been executed.

## What this is

A **generality check**, not a new primary result. We run UCF (Yan et al., ICCV
2023) through our own evaluation protocol to test whether the effects we measure
are specific to one architecture or recur across them.

UCF is **supplementary**, not a Target and not the course's Support role:

| role | paper |
|---|---|
| Baseline | Xception, CVPR 2017 |
| Target | F3-Net, ECCV 2020 |
| Support | Kapoor & Narayanan, *Patterns* 2023 |
| **Supplementary** | **UCF, ICCV 2023 — this document** |

❌ It is **not** an attempt to beat anything, and **not** a benchmark comparison.
We make no claim that UCF is better or worse than Xception. The quantity of
interest is how the *interval* behaves, not where the point estimate sits.

## Why this campaign and not another

| option | cost | blocker |
|---|---|---|
| **UCF, 5 seeds, fixed c40 video-disjoint split** | ~5 GPU-h | **none — chosen** |
| V2 replication (seen vs unseen) | ~1 GPU-h | needs the V2 manifests, uncommitted (recovery runbook gap 1) |
| L1/L2 protocol gap for UCF | ~10 GPU-h | two full campaigns |

The chosen option has **no prerequisite**: five Xception seeds already sit on that
exact split with dumps committed, and the split is verified against
`results/reference/ffpp_c40_vid_seed0_test_split.json`. Five UCF runs complete a
second model pair.

## What five runs unlock

Three replications, all on analysis code that already exists and needs no edits:

1. **Unit of evaluation** — UCF − Xception at frame / video / component. Does the
   frame bootstrap exclude zero while clustered units do not, as it did for FAD?
2. **Estimand** — the same difference under frame-pooled, mean-logit, mean-probability.
3. **Conditional vs crossed** — both variance sources, as in the primary analysis.

## 🚨 Configuration: port UCF into OUR recipe

128 px, batch 64, AdamW `lr=3e-4` `wd=0.05`, cosine, 15 epochs, focal loss
(γ=2, α from train class balance), fp32, checkpoint = best validation ROC-AUC.

UCF's published config uses batch 16 at its own resolution. **Running theirs would
confound architecture with training setup — the exact error this paper exists to
identify.** We would be unable to use the result.

✅ Record the departure from their published setup as a stated limitation.
❌ Do not run their config and "note the difference".

## Interpretation rule — fixed before running

| outcome | what we report |
|---|---|
| Unit finding replicates (frame excludes zero, clustered units do not) | The effect is **not architecture-specific**. Report alongside the primary, clearly labelled supplementary |
| Unit finding does not replicate | Report it. **Architecture-dependence is a finding**, not a failure, and it bounds our generality claim |
| UCF fails to converge at 128 px | Report that too. A modern detector failing under a reduced budget is a limitation of our study, stated plainly, not a hidden result |

❌ Under no outcome does this change `canonical.json` or the primary FAD conclusion.

## 🚨 Isolation — how the existing results stay untouched

1. **Branch.** All work off `main` until it passes.
2. **New directories only.** `results/predictions_ucf/`, `results/in_domain_c40_ucf/`.
   Never into `predictions/`, `predictions_v8/`, `predictions_v2/`.
3. **Do NOT touch `src/train.py`.** UCF's multi-term loss will tempt an edit to the
   training loop. Write **`src/train_ucf.py`** as a separate entry point instead.
   `results/reference/train_args_c40_vid.json` records that the effective settings
   are the `src/train.py` default **at commit `e68b559`** — that file's contents are
   load-bearing provenance.
4. **`config.py` additive only.** A new `ucf` branch in `build_model` is fine;
   modifying an existing branch is not.
5. **Analysis code unchanged.** `crossed_boot.py`, `agg_space.py` and
   `cluster_boot.py` take globs and run on new paths as-is. Editing one is a signal
   something has gone wrong.
6. **Prove it, do not assume it.** Before merging:

```bash
git diff main -- src/train.py src/crossed_boot.py src/agg_space.py \
                 src/cluster_boot.py src/clusters.py     # must print nothing
python -m src.reproduce                                   # must return 16/16
```

Empty diff plus 16/16 means the code that generates existing results and the data
it reads are both provably unchanged.

## Run guards — from this project's own failures

- **Dump predictions after every run.** C1 and C2 are permanently aggregate-only
  because an earlier campaign saved checkpoints and not scores.
- **Distinct `--out-dir` per run**, then check for **5 distinct md5 hashes**. Two
  prediction calls once shared a path and one silently overwrote the other.
- **Verify the split** against the committed reference JSON *before* training.
- **No `--amp`.** Passing it once cost a 155-minute run and confounded precision
  mode with split policy.

## Risks, named now

UCF was designed for larger inputs and may converge poorly at 128 px. Its loss has
several terms, so integration is real work rather than a config change. Five runs of
a heavier model may exceed the ~5 GPU-hour estimate.

None of these threaten the primary result, because of the isolation above.

## Reporting

A separate section, clearly labelled supplementary. The Xception vs Xception+FAD
comparison remains the primary result throughout, and UCF is never merged into it.
