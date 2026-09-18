# Night-1 batch — B1, B2, C1, C2 (2026-09-18)

Run unattended on Kaggle, `04:13–08:27` (**4 h 14 m**, under the 5.1 h estimate).
12 of 13 stages returned 0. Driver `docs/night1_run.py`, governed by
`docs/c1-lr-prespecification.md` and `docs/c2-repeat-prespecification.md`, both
committed before the run.

**Split verified before any training:** *"split VERIFIED against reference: 30 test
targets match"*, `train/val/test = 24000/3000/3000`.

⚠️ Nothing here changes `results/canonical.json`.

---

## C2 — repeat audit: the substantive result

Three runs of a **nominally identical** configuration (fixed split 0, training
seed 0, 15 epochs, fp32):

| run | Xception | + FAD | FAD − Xception |
|---|---|---|---|
| V8 seed 0 (earlier session) | 0.79705 | 0.81104 | **+0.0140** |
| `c2_rep1` (this session) | 0.8043 | 0.7929 | **−0.0114** |
| `c2_rep2` (this session) | 0.8193 | 0.7882 | **−0.0311** |

| quantity | value |
|---|---|
| within-session spread (rep1 vs rep2) | **0.0197** |
| full range across all three | **0.0451** |
| **sd across 3 nominally identical runs** | **0.0226** |
| V8 sd across **5 different seeds** | **0.0145** |
| earlier cross-session movement | 0.0336 |
| reference effect | 0.0140 |

### Against the prespecified decision rule

The rule keys on within-session spread against the earlier 0.0336 movement.
**0.0197 is 59% of 0.0336 — the same order of magnitude**, so the first branch
fires:

> The earlier movement is **consistent with ordinary run-to-run variation**. No
> session-level effect is indicated.

The third branch also applies, on a different quantity:

> **Run-to-run variability at this scale is larger than previously estimated.**

**Re-running the identical configuration produced sd 0.0226, larger than the
0.0145 obtained by varying the training seed.** The seed is not the unit of
variation — the same seed does not reproduce itself. The V8 fixed-split figure of
0.0145 **understates** run-to-run variability and must be reported as a lower
figure from a smaller set of repeats, not as the variability of the pipeline.

⚠️ **n = 3.** Not an estimate of session-level variance. **No causal attribution**,
per the prespecification, which binds even though the result now looks systematic.

---

## C1 — learning-rate sweep: 1 of 3 cells usable

| lr | Xception | + FAD | difference | best epoch (x/f) | status |
|---|---|---|---|---|---|
| 1e-4 | 0.7790 | 0.7854 | **+0.0064** | 2 / 2 | ✅ **VALID** |
| 3e-4 (reference) | 0.8087 | 0.8238 | +0.0151 | **4 / 4** | ⚠️ **INCONCLUSIVE** — both peaked at the final epoch |
| 1e-3 | **0.5000** | 0.7415 | +0.2415 | 0 / 4 | ⛔ **DIVERGED** |

**The only valid cell gives +0.0064, inside the frozen crossed interval
[−0.0358, +0.0180].** On that cell, the null is not attributable to the learning
rate.

### ⚠️ Two problems, both declared rather than worked around

**1. The prespecified epoch rule caught two of three cells.** It states that a cell
peaking at the final epoch is inconclusive. At 5 epochs (0–4) the reference-lr cell
peaked at epoch 4 in **both** arms. The 5-epoch budget was justified in advance by
c40 best-epoch averaging ≈5 — **that justification was wrong for this purpose**.
The rule worked; the budget was too tight. A rerun at 15 epochs would be needed to
make the reference cell interpretable.

**2. At lr 1e-3 the Xception arm did not train.** `val_auc` is **exactly 0.5000**
at every epoch, `train_loss` frozen at 0.0555, `val_acc` alternating 0.8/0.2 — the
model predicts a single class. The +0.2415 is an artifact of a **failed baseline**,
not an architectural difference.

**The prespecification did not anticipate divergence.** Read literally, its rule
*"at any lr, FAD − Xception exceeds +0.0180 → the result is lr-sensitive"* is
triggered by this cell. That reading is false on the facts. But **adding a
divergence-exclusion criterion now would be post-hoc**, so it is declared here as a
gap in the prespecification rather than quietly applied:

> The lr 1e-3 cell triggers the prespecified lr-sensitivity branch. We report that
> it does, and that the Xception arm reached chance performance with a frozen
> training loss, so the cell measures a training failure rather than an
> architectural contrast. Excluding it is a judgment the prespecification does not
> license.

**Conservative reading:** one valid cell, +0.0064, inside the crossed interval.
Two cells uninterpretable. The lr objection is **partially** answered — for 1e-4
only.

---

## B2 — V2 seen/unseen: ⛔ INVALID — trained on the wrong data

Manifests built correctly and matched exactly:

```
seen_eval   {'crops': 750, 'videos': 150, 'groups': 30, 'reals': 150,
             'by_manipulation': {Deepfakes 150, Face2Face 150, FaceSwap 150,
                                 NeuralTextures 150, real 150}}
unseen_eval {identical shape}
train 24000 -> 23250 crops
```

A model trained and reported `test_auc = 0.8061`. **It is not the V2 model.**

`train.py` resolves its manifest as
`out_root/manifests/<dataset-name>_seed<n>_sz<n>.csv`. The driver wrote the V2
manifests to `night1/manifests/` but trained with `--out-root night1/v2`, so
`train.py` looked in `night1/v2/manifests/`, found nothing, and **silently
regenerated a default split**.

**The evidence is in the log.** `seen_unseen` reported `train 24000 -> 23250
crops`; the training run reported `train=24000`. It trained an ordinary L2 model on
the standard split — effectively a duplicate of an existing run — not the
seen-manifest model.

**And the driver never scored anything against the UNSEEN manifest**, so even a
correct model would have produced no leakage estimate.

Two bugs, both mine. **B2 must be re-run in full** (~85 min). Fixed in
`docs/night1_run.py`: `--out-root` is now `OUT` for both stages, and two
`src.predict` calls score the one trained checkpoint against **both** manifests —
`predict.py` takes the checkpoint from `--run` and the manifest from
`--dataset-name`, so it supports exactly this.

⚠️ **This is the failure mode the split guard exists to prevent, in a place the
guard did not cover.** The guard verifies the *base* c40 manifest before training;
it does not verify that each training stage then consumed the manifest intended for
it. A silent regeneration produced plausible numbers from the wrong data.

---

## B1 — failed twice; it cannot run first at all

### Second attempt, 2026-09-18

```
skip ffpp_c40_vid_xception_seed0: no checkpoint
KeyError: 'delta_auc'
```

**This is structural, not a flag mistake.** `band_ablation` re-scores a *trained
model* with frequency bands masked, so it needs `results_root/<run>/best.pt`.
**Zero `.pt` files are committed** — checkpoints are too large — so in a fresh
container no checkpoint exists for a run trained in an earlier session. Only that
run's predictions were kept.

Two fixes applied:

1. **B1 now runs after B2**, against the checkpoint B2 produces
   (`ffpp_c40_v2seen_xception_seed0`). A different model from the original target,
   but B1 supports no claim — it exists to demonstrate the script executes.
2. **`band_ablation` now exits with a clear message** instead of `KeyError:
   'delta_auc'` when every run is skipped. An empty frame crashing on a pivot told
   us nothing; the new message names the checkpoint requirement.

### First attempt, 2026-09-18 (earlier)

```
FileNotFoundError: .../manifests/ffpp_seed0_sz128.csv
```

`band_ablation.py` took its default `--dataset-name ffpp`, so it looked for
`ffpp_seed0_sz128.csv` instead of `ffpp_c40_vid_seed0_sz128.csv`. The runbook
flagged this argument as an untested guess and marked its failure ignorable. It
supports no claim.

Fixed: `--dataset-name ffpp_c40_vid --seed 0` are now passed, and the run name no
longer carries a `_test` suffix.

**This failure was benign precisely because it was loud.** B2's was not.

---

## What must be re-run

| | action | cost |
|---|---|---|
| **B2** | **re-run in full** — the training used the wrong split and nothing was scored against the unseen manifest | ~85 min |
| **B1** | re-ordered to run after B2 against its checkpoint. **Not a 5-minute fix** — it needs a trained model in the same session, and no checkpoints are committed | free, rides on B2 |
| **C1** | ⛔ **PROHIBITED.** `docs/c1-lr-prespecification.md` §6 forbids *"extending the epoch budget for cells that look unfinished"*. Two cells are uninterpretable and must be reported as such | — |
| **C2** | nothing. Complete and analysed | — |

C1 is the discipline working against us, which is the point of writing it down
first. The reference learning rate is in any case already covered at 15 epochs by
the five V8 runs, so the inconclusive cell costs little.

## Owed on arrival of the tarball

1. **Verify test-item identity** between the `c2_*` runs and the V8 seed-0 run.
   They used different `--dataset-name` tags, so manifests were written under
   different names. `make_splits` is deterministic given (items, seed, group_by)
   and the counts match, but identity must be **checked from the dumps**, not
   assumed — the same class of silent failure the split guard exists to prevent.
2. Commit the prediction CSVs.
3. Recompute every number above from the dumps rather than from the log.
