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

---

# Re-run results — 2026-09-18, 4h37m

The driver executed **three times** (the `--stages` flag never reached the run
cell, and the second clone failed with *"destination path already exists"*, so all
three invocations used the 13:47 clone — **without** the B1 reorder or the
prediction-collision fix).

## 1. B2 — the V2 seen condition is extreme

| run | seen test AUC |
|---|---|
| 14:21 | **0.9826** |
| 18:23 | **0.9884** |

The model was trained on the seen manifest and evaluated on **held-out crops from
videos it trained on**. It scores **0.98–0.99** — essentially L1's 0.9933 — where
the same architecture under video-disjoint evaluation scores **0.8169**.

⚠️ **This is not yet the V2 estimate.** V2 compares seen against unseen **from the
same model**; the number above compares a seen score against a *different* model's
L2 score. The unseen dump exists (written second, so it survived the collision) and
the comparison must be computed from the two CSVs, not from these logs.

**But the direction is already informative.** A seen-video condition reaching 0.99
is difficult to reconcile with the ~18-point protocol gap being mostly partition
difficulty.

## 2. C1 — the replication flipped sign at both usable rates

| lr | 2026-09-17 | re-run | |
|---|---|---|---|
| 1e-4 | **+0.0064** | **−0.0084** | ⚠️ **sign flip** |
| 3e-4 | **+0.0151** | **−0.0062** | ⚠️ **sign flip** |
| 1e-3 | +0.2415 | +0.0457 | same sign, both degenerate |

And in the re-run, **five of six cells peaked at the final epoch** — including
lr 1e-4, the only cell the first sweep certified valid.

**C1 is uninformative, and the replication is what establishes that.** Two sweeps
at identical settings disagree in sign at both interpretable rates. No conclusion
about learning-rate sensitivity can be drawn from either.

✅ **One prespecified question is answered.** `docs/night1-rerun-handling.md` asked
whether the lr 1e-3 divergence reproduces. **It does — Xception reached exactly
0.5000 with a frozen training loss in both sweeps.** Divergence at that rate is a
property of the configuration, not a one-off failure.

⚠️ The two sweeps are **not pooled**, per the handling rule.

## 3. C2 — five observations of one nominal configuration

Same split, same seed, same code, same hyperparameters, fp32 throughout.

| run | session | Xception | + FAD | difference |
|---|---|---|---|---|
| V8 seed 0 | A | 0.79705 | 0.81104 | **+0.0140** |
| `c2_rep1` | B | 0.8043 | 0.7929 | **−0.0114** |
| `c2_rep2` | B | 0.8193 | 0.7882 | **−0.0311** |
| `c2_rep1` re-run | C | 0.7949 | 0.8178 | **+0.0229** |
| `c2_rep2` re-run | C | 0.7907 | 0.8065 | **+0.0158** |

Signs: **+ − − + +**. Range **0.0540** = **3.9×** the reference effect.

### What the handling rule permits

The permitted use is the **cross-session comparison** the original prespecification
wanted and could not make:

> **Session B produced two negative differences; session C produced two positive
> differences, from the same nominal configuration.** Within each session the two
> repeats agree in sign; across sessions they do not.

⚠️ **Not pooled.** The sd **0.0226** across the three 2026-09-17 observations
stands as reported and is **not** recomputed with the re-run.
⚠️ **No causal attribution**, per `docs/c2-repeat-prespecification.md` §5 — and the
rule binds precisely because the pattern now looks systematic. cuDNN kernel
selection, hardware allocation and library versions are untested candidates.

### Why this matters most

A nominally identical configuration produced a FAD − Xception difference of **both
signs**. Any architectural claim resting on a single training run is therefore
underdetermined at this scale — which is contribution 3's thesis, demonstrated on
our own pipeline rather than argued.

## 4. B1 — failed in all three invocations

Same cause each time: no checkpoint for `ffpp_c40_vid_xception_seed0`, since none
are committed. Both fixes (reorder after B2; clean exit instead of `KeyError`) were
pushed after this clone and are untested.

## Owed

1. **Compute the V2 estimate** from `night1/v2_predictions/` — the surviving CSV is
   the **unseen** one; the seen dump was overwritten by the collision. Re-score from
   the checkpoint, which is in the version output.
2. **Regenerate C1/C2 prediction dumps** while the checkpoints exist. All ten runs
   are aggregate-only.
3. None of the above changes `results/canonical.json`.

---

## ⚠️ Both C1 and C2 conclusions reverse between executions

External review read session C's C2 repeats and sweep 2's C1 in isolation, and
reached conclusions that the other execution contradicts. Recorded because the
same reading is easy to arrive at from any single log.

| claim, from one execution | reversed by |
|---|---|
| *"Both repeats favoured FAD"* (+0.0229, +0.0158) | session B, four hours earlier: **−0.0114, −0.0311** |
| *"FAD did not outperform Xception at either viable lr"* (−0.0084, −0.0062) | sweep 1: **+0.0064, +0.0151** |

**Neither sweep nor session supports a directional claim.** The instability *is*
the result, and it is only visible with both executions side by side.

### Safe wording

✅ *"Two nominally identical in-session repetitions produced FAD − Xception
differences of +0.0229 and +0.0158; two further repetitions of the same
configuration in a different session produced −0.0114 and −0.0311. The recorded
seed and split did not preserve the sign of the architectural difference."*

✅ *"In two one-seed, five-epoch sensitivity sweeps at identical settings, the
FAD − Xception difference changed sign at both non-divergent learning rates. The
sweep does not support a directional claim."*

❌ Any sentence of the form *"FAD did/did not outperform Xception at lr X"* citing
one sweep.

## Engineering defects, from the same review — both fixed

| defect | fix |
|---|---|
| Dry-run summary printed **OK** for steps that never executed | now prints **DRY-RUN (not executed)** |
| Both B2 scorings reported OK while one overwrote the other | `verify_distinct_dumps()` now hashes both files and **aborts if identical or missing** |

A zero return code records that a process finished, **not** that it produced a
distinct artifact. That gap is what made the collision invisible for two runs.

### Also noted, not yet fixed

- The notebook cloned into a non-empty directory, so the second and third driver
  invocations silently used the **13:47 clone** — without the B1 reorder or the
  collision fix. Future runs should clone fresh or `git pull`.
- **B2 trained twice under the same run name**, so the 18:23 checkpoint replaced the
  14:21 one. The surviving checkpoint corresponds to **seen AUC 0.9884**.
