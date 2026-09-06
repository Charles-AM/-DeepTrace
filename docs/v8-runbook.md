# V8 runbook — fixed split, varying training seed (Save & Run All)

Separates **optimisation variability** from **split composition**. Every earlier
multi-seed result varies both at once: `make_splits` seeds its permutation with
`--seed`, so each seed drew a different partition. V8 freezes one split and varies
only training randomness. The difference between the two standard deviations is
the split-composition contribution — the last of the paper's four rungs.

**Batch mode.** Written for *Save Version → Save & Run All*. Close the laptop.

| | |
|---|---|
| Accelerator | **GPU T4 ×2** |
| Internet | **On** (clone) |
| Runs | 10 = 2 configs × 5 training seeds, c40, `--split-seed 0`, 15 epochs |
| Estimated | ~6 h (session limit 12 h) |

## Inputs to attach

**Required — one input:** the notebook output **`c40-run`** (under
`charlesappiahmanu`). It contains `ffpp_c40_crops/`, which is where every c40
result's crops actually live. Add Input → Notebook Output → search `c40-run`.

**Optional:** the c40 video-level notebook output, if you can find it. It supplies
`results/manifests/ffpp_c40_vid_seed0_sz128.csv` and the script will copy that
verbatim. Without it the script regenerates the split from the same crops —
`make_splits` is deterministic given (items, seed, group_by), so the result is the
same partition.

**Either way the split is verified, not assumed.** The seed-0 test membership is
recorded in `results/reference/ffpp_c40_vid_seed0_test_split.json` (30 target
sequences, extracted from the committed prediction dumps). The script compares the
manifest it ends up with against that list and aborts on any difference. A split
that quietly differed would make V8 incomparable with the varying-split runs it is
subtracted from — the one failure that would invalidate the experiment while
raising no error at all.

## Design decisions worth knowing

**Runs are interleaved by seed, not grouped by config** — `(s0 xception, s0 fad),
(s1 xception, s1 fad), …`. If the session dies at seed 3 you still have complete
*pairs* for seeds 0–2, which is analysable. Grouped by config, a death midway
would leave five baselines and no comparisons.

**Predictions are dumped after each pair**, so partial completion still yields
usable analysis rather than only checkpoints.

**The seed-0 manifest is copied, not regenerated.** Regenerating is deterministic
and *should* reproduce it, but copying makes the split byte-identical to the runs
V8 is compared against, which removes the question entirely.

**Data ordering and augmentation are left free.** They vary with the training
seed and are legitimately part of training-process variability. Pinning a shared
generator across configs would couple the data stream, shrink the paired
difference, and make V8's sd non-comparable with the varying-split sd of 0.0188 it
is subtracted from.

**A time guard stops launching new runs after 10.5 h**, so the packaging step
always runs. Nothing is lost to a hard cutoff.

---

## Cell 1

```python
!cd /kaggle/working && rm -rf ./-DeepTrace && git clone -q https://github.com/Charles-AM/-DeepTrace.git ./-DeepTrace
!cd /kaggle/working/-DeepTrace && git log --oneline -1
```

## Cell 2 — the whole run

Paste `docs/v8_run.py` (in the repo) as a `%%writefile`, or simply:

```python
!cd /kaggle/working/-DeepTrace && python docs/v8_run.py
```

## Cell 3 — package

```python
!cd /kaggle/working && tar czf v8_results.tar.gz results preds && du -h v8_results.tar.gz
```

⚠️ **Everything must land in `/kaggle/working`.** The V1 notebook's outputs were
not retained and its prediction dumps were lost; only the rendered HTML survived.
Cell 3 is what makes this run recoverable.

## After it finishes

```
python -m src.paired_summary        # sd across seeds, fixed split
```

Compare that sd against the varying-split sd of **0.0188** (c40, n=5). The
difference is the split-composition contribution. Report both, and state plainly
that the varying-split figure is an upper bound on optimisation variability.
