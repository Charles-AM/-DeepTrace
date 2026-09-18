# What gets committed, and what does not

Adopted 2026-09-18, after two failures traced to the same cause: **the substrate
was discarded and only the summary kept.**

## The rule

| artifact | size | committed? |
|---|---|---|
| **Per-prediction CSV** | ~650 KB | ✅ **always, for every run** |
| Per-run provenance JSON | ~1 KB | ✅ always |
| Analysis outputs, READMEs | small | ✅ always |
| Manifests | ~2 MB | ❌ regenerable — `make_splits` is deterministic given (items, seed, group_by), and every use is verified against `results/reference/` |
| **Model checkpoints** (`best.pt`) | **~83 MB** | ❌ **never to git** — see below |

## Why checkpoints do not go in git

| | |
|---|---|
| one checkpoint | **83 MB** (model state only; with optimizer state it would be ~250 MB) |
| GitHub **hard** limit per file | **100 MB** — a push above this is **rejected** |
| one 10-run campaign | **~0.8 GB** |
| current repo | **10 MB** |

A single campaign would inflate the repo ~80×, and every file would sit at 83% of
a hard limit that a larger config would breach outright. Git is the wrong store.

**Instead:** checkpoints stay in the Kaggle version output, and the repo records
**where** and **which** — see the registry below.

## Why per-prediction dumps always go in git

This is the one that has cost us twice.

`train.py` writes a per-run JSON and `best.pt`. It does **not** write per-frame
scores — `src.predict` is a separate step. Skip it and the run is stuck at
aggregate level forever.

**Two runs are already in that state:**

- **The entire L1 campaign** (`results/in_domain_c40/`) — aggregate metrics only,
  so the L1 column of the protocol-gap table cannot carry a cluster bootstrap, an
  interval, or an independent recomputation (contribution 1 evidence map §0).
- **Last night's C1 and C2 runs** — including the sd 0.0226 repeat-audit result,
  the strongest finding of the two nights.

A dump is **650 KB against an 83 MB checkpoint**. There is no circumstance in which
skipping it is the right trade.

**`docs/night1_run.py` now calls `dump_predictions()` after every successful
training run.** Any future driver must do the same.

## Checkpoint registry

Because the bytes are not in git, record the location so a checkpoint can be found:

| run | campaign | source | notes |
|---|---|---|---|
| `ffpp_c40_v2seen_xception_seed0` | night-1 re-run, 2026-09-18 | Kaggle version output | needed to re-score V2 seen/unseen |
| `c1_lr*_{xception,xception_fad}_seed0` | night-1, 2026-09-18 | Kaggle version output | needed to regenerate C1 dumps |
| `c2_rep{1,2}_{xception,xception_fad}_seed0` | night-1, 2026-09-18 | Kaggle version output | **needed to regenerate C2 dumps — the sd 0.0226 result** |
| V8 campaign (10 runs) | 2026-09-06 | not retained | dumps were committed, so this is fine |

⚠️ **Kaggle version outputs are not permanent.** Retrieve anything in the table
above that is still needed, or accept that the run becomes aggregate-only.

## Recovery, while the checkpoints still exist

Attach the night-1 version output to a notebook and run `src.predict` per run. It
is inference only — a few minutes, no retraining. Cells in `docs/b2-rescore.md`
show the pattern; only `--run` and `--dataset-name` change.

## Milestones

Commit per milestone, not in one batch at the end — a result that exists only in a
session that later dies is not a result.

| milestone | commit |
|---|---|
| Prespecification | **before** the run it governs |
| Runbook + driver | before the run, dry-run validated |
| Prediction dumps | as soon as retrieved |
| Analysis + README | with the numbers recomputed from the dumps, not from the log |
| Corrections | immediately, with the reason in the message |
