# Prespecification — UCF matched-recipe evaluation (NOT YET RUN)

**Written 2026-10-27, revised 2026-10-27 after review. Before any run.** Git
history is the evidence. Nothing here has been executed.

## What this is, and what it is not

A **matched-recipe evaluation of UCF (Yan et al., ICCV 2023) as a second
architectural case study**, to test whether our evaluation findings depend on the
specific frequency component or recur across architectures.

❌ **Not a replication of UCF.** We do not reproduce their training setup,
resolution, dataset scale or reported result. Their published FF++ figure is 0.986
at c23 on the full dataset; ours will be at 128 px on 150 pairs at c40. **Never
compare the two numbers.**

❌ **Not a benchmark comparison.** We make no claim that UCF is better or worse
than Xception. UCF − Xception is a difference whose *interval behaviour* we study;
which model wins is irrelevant to the question.

❌ **Not required to complete the paper.** FAD remains the primary case study. Run
this only if the implementation validates without touching the frozen pipeline.

| role | paper |
|---|---|
| Baseline | Xception, CVPR 2017 |
| Target | F3-Net, ECCV 2020 |
| Support | Kapoor & Narayanan, *Patterns* 2023 |
| **Supplementary** | **UCF, ICCV 2023 — this document** |

## Why this campaign

| option | cost | blocker |
|---|---|---|
| **UCF, 5 seeds, fixed c40 video-disjoint split** | ~5 GPU-h | **none — chosen** |
| V2-style seen/unseen with UCF | ~1 GPU-h | needs V2 manifests, uncommitted |
| L1/L2 protocol gap for UCF | ~10 GPU-h | two full campaigns |

Five Xception seeds already sit on that exact split with dumps committed, verified
against `results/reference/ffpp_c40_vid_seed0_test_split.json`. Five UCF runs
complete a second model pair.

## Three generalisation checks (not replications)

Whether each finding **also appears** with another architecture:

1. **Resampling unit changes the uncertainty** — frame / video / component
2. **Aggregation rule changes the measured difference** — frame-pooled / mean-logit / mean-probability
3. **Conditional and crossed analyses support different conclusions**

All run on existing analysis code, which needs no edits.

---

## 🚨 What "matched recipe" means, precisely

**Swapped to ours** — optimiser (AdamW `lr=3e-4`, `wd=0.05`), cosine schedule,
15 epochs, batch 64, 128 px, fp32, our data / split / augmentation, checkpoint =
best validation ROC-AUC, classification term = focal loss (γ=2, α from train class
balance).

**Retained from UCF** — every module and auxiliary objective the method needs to
*be* UCF: the disentanglement encoder, the decoder and its reconstruction loss, the
multi-task head, the contrastive regulariser. Remove any of these and it is no
longer UCF.

### ⚠️ The leading risk: inherited loss weights

UCF's total objective combines classification with reconstruction, multi-task and
contrastive terms, each carrying a relative weight. **Those weights were set at
batch 16 and their resolution.** At batch 64 and 128 px they are likely wrong, and
the auxiliary terms will either dominate or vanish.

This is the most probable cause of a convergence failure, and keeping the modules
intact does not fix it. We are inheriting hyperparameters tuned for a configuration
we are not running — in a study whose premise is that we do not tune.

**We will not tune them.** We run their published weights, report what happens, and
state this as a limitation. Tuning them would be the one thing this paper cannot do.

## 🚨 Implementation fidelity — all checks pass BEFORE the five runs

**Known-answer test.** Run our own Xception through `train_ucf.py` with the UCF
modules disabled. It must reproduce the committed seed-0 AUC of **0.81347**. The
answer is already on disk, which makes this stronger than any tensor inspection.

Then:

- Baseline path reproduces our existing Xception input/output format
- UCF's auxiliary losses receive the tensors they expect
- One small batch completes forward **and** backward
- Predictions are **not constant** — a collapsed model gives AUC ≈ 0.5 and looks
  like a convergence failure when it is a wiring bug
- Gradients reach the UCF-specific modules, not just the backbone
- Parameters, FLOPs and latency recorded via `eff_table.py` — UCF may cost
  considerably more than FAD's +0.24% / +2.70% / +3.19%

## Sanity band — fixed in advance

Our Xception sits at ≈0.81 at c40.

| UCF test AUC | action |
|---|---|
| ≈ 0.5 | **broken wiring.** Investigate, do not report |
| 0.75 – 0.90 | plausible, proceed |
| ≈ 0.99 | **a leak.** Investigate, do not report |

Stating the band now stops a suspicious number being rationalised later.

## Interpretation rule — fixed before running

| outcome | what we report |
|---|---|
| Checks 1–3 also appear with UCF | The effects are **not architecture-specific**. Report alongside the primary, clearly labelled supplementary |
| They do not appear | Report it. **Architecture-dependence is a finding**, and it bounds our generality claim |
| UCF fails to converge | Report plainly. A modern detector failing under a reduced budget and inherited loss weights is a limitation of our study, not a hidden result |

❌ No outcome changes `canonical.json` or the primary FAD conclusion. A failed or
unstable result stays supplementary and **does not weaken the validated FAD
findings**.

## 🚨 Isolation — prove it, do not assume it

1. **Branch.** All work off `main` until it passes.
2. **New directories only.** `results/predictions_ucf/`, `results/in_domain_c40_ucf/`.
3. **Do NOT touch `src/train.py`.** UCF's multi-term loss will tempt an edit. Write
   **`src/train_ucf.py`** instead. `results/reference/train_args_c40_vid.json`
   records the effective settings as the `src/train.py` default **at commit
   `e68b559`** — that file's contents are load-bearing provenance.
4. **`config.py` additive only.**
5. **Analysis code unchanged.** It takes globs and runs on new paths as-is.
6. **Before merging:**

```bash
git diff main -- src/train.py src/crossed_boot.py src/agg_space.py \
                 src/cluster_boot.py src/clusters.py     # must print nothing
python -m src.reproduce                                   # must return 16/16
```

## Run guards — from this project's own failures

- **Dump predictions after every run.** C1 and C2 are permanently aggregate-only
  because an earlier campaign kept checkpoints and not scores.
- **Distinct `--out-dir` per run**, then verify **5 distinct md5 hashes**.
- **Verify the split** against the committed reference JSON before training.
- **No `--amp`.** Passing it once cost a 155-minute run.

## Reporting

A separate section, labelled supplementary. The Xception vs Xception+FAD comparison
remains the primary result throughout, and UCF is never merged into it.
