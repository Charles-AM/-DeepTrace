# Prespecification — NPR-inspired supplementary campaign (NOT YET RUN)

**Written 2026-10-31, before any run.** Git history is the evidence.

Replaces the UCF and Xception+D plans
(`docs/ucf-supplementary-prespecification.md`, `docs/xception-d-supplementary-plan.md`),
which are **parked** — UCF's reconstruction objective requires balanced paired
batches, which would change the class balance our models train under and confound
the comparison.

## Purpose

Test whether the aggregation and uncertainty findings extend beyond FAD, using a
second **deterministic input transform** rather than a second learned subsystem.

FAD remains the primary case study. This is supplementary throughout.

## Framing — fixed wording

> We conduct an **NPR-inspired matched-recipe evaluation using Xception as the
> shared backbone**. NPR adds a fixed local pixel residual transform while the
> data, backbone, loss, sampler, optimiser, training budget and evaluation remain
> unchanged.

❌ **Not a reproduction of NPR.** The original uses a ~1.44M-parameter custom CNN
trained on ForenSynths with Adam at batch 32. We replace that classifier with
Xception and train on FF++.
❌ **No exclusion test.** NPR reports no matched FF++ comparison of Xception
against Xception+NPR, so there is **no published reference effect** to test against.
This study asks whether our *evaluation effects* recur, not whether a claim reproduces.

## The transform — verified from source

`networks/resnet.py`, chuangchuangtan/NPR-DeepfakeDetection:

```python
def interpolate(self, img, factor):
    return F.interpolate(F.interpolate(img, scale_factor=factor, mode='nearest',
        recompute_scale_factor=True), scale_factor=1/factor, mode='nearest',
        recompute_scale_factor=True)

NPR = x - self.interpolate(x, 0.5)
x = self.conv1(NPR * 2.0/3.0)
```

| property | value |
|---|---|
| learned parameters | **zero** |
| output shape | **unchanged** — 3 channels, 128×128 |
| backbone modification | **none** — not even `in_chans` |
| auxiliary losses | none |
| batch-structure requirement | **none** (checked in code, not inferred from the paper) |

🚨 **Placement.** Their code computes the residual *inside* `forward`, so **after**
dataloader normalisation and immediately before `conv1`. Applying it before
normalisation gives a per-channel scaled version, not the same operation.

## Scope — TEN runs, both arms retrained

🚨 **Revised 2026-10-31.** The original plan paired five new NPR runs against the
committed V8 Xception runs. That is unsound.

`results/in_domain_c40_fixedsplit/README.md` records a nominally identical seed-0
configuration repeated across sessions: the FAD−Xception difference moved from
−0.0196 to **+0.0140**, a movement of **0.0336 — 2.4× the reference effect — and
it flipped sign.** The two arms moved in *opposite* directions, so this is not a
constant offset that cancels in a paired difference.

The primary FAD comparison does not have this problem: V8 trained both arms in one
campaign. Reusing half of it for NPR would give the supplementary study a weaker
property than the primary has.

**Therefore: five new plain Xception runs and five new Xception+NPR runs, both
arms in the same campaign.**

| experiment | covered |
|---|---|
| 3 — aggregation | ✅ |
| 4 — units, conditional vs crossed | ✅ |
| 1 — splitting | ❌ second campaign |
| 2 — prior exposure | ❌ manifests uncommitted |

### 🚨 Counterbalanced run order — committed before training

Always training Xception first would let an **order effect** load onto one arm.

| seed | first | second |
|---|---|---|
| 0 | Xception | NPR |
| 1 | NPR | Xception |
| 2 | Xception | NPR |
| 3 | NPR | Xception |
| 4 | Xception | NPR |

❌ Do **not** attribute any observed movement to heat, caching or memory
fragmentation. Those mechanisms have not been tested. Counterbalancing controls a
*possible order effect*; the mechanism is unknown.

### Session rule

**Each Xception/NPR seed pair must complete within one session.** If a session ends
between the two arms, the completed run stays recorded but is **excluded from the
primary paired analysis**, and both arms for that seed are rerun together.

Ideally all ten runs share one session. At minimum each matched pair does.

Record per run: session identifier, start time, model order, seed, device,
software versions, checkpoint path.

### 🚨 Analysis gate — assert, do not assume

The analysis is glob-driven. A pattern catching `results/predictions_v8/` alongside
`results/predictions_npr/` would **silently** mix sessions — the same failure mode
as the night-1 output collision.

Before any analysis, assert: exactly the five new same-session Xception and five
NPR dumps, one per seed, identical component membership across all ten.

❌ **The committed V8 Xception runs are never mixed into the NPR effect estimate.**

### Bonus: the repeatability audit upgrades from n=1 to five paired seeds

Comparing the five new Xception runs against the five committed V8 Xception runs
gives a five-seed paired cross-session contrast, where the current audit is one
observation.

**Fixed wording:**

> We repeated five seed configurations in a second campaign using the same fixed
> split, recipe and protected training code. This expands the audit from one
> repeated seed configuration to five. Because all comparisons involve the same two
> campaign sessions, it does not estimate the distribution of session-level
> variation.

Report: the five paired AUC changes, their signs, mean and range, the largest
absolute movement, and whether any architectural or threshold interpretation would
change.

❌ **No formal session-level confidence interval.** Five observations sharing one
session pair cannot estimate session variance.
⚠️ Five paired *seed configurations*, **one** contrast between two campaign
sessions. Never call the session sample size five.

## 🚨 Directional prediction — written before running

NPR targets artifacts from **learned upsampling**. The four FF++ methods differ:

| method | mechanism | NPR expected to help |
|---|---|---|
| Deepfakes | two autoencoders, shared encoder | **most** |
| NeuralTextures | learned neural texture + rendering network | plausibly |
| FaceSwap | classical graphics: landmarks, 3D template, blending | least |
| Face2Face | 3D reconstruction + image-based rendering | least |

**Prediction, with the effect defined explicitly:**

> Using **video-aggregated mean-logit AUC**, we predict that the **NPR minus
> Xception difference** will be largest for Deepfakes, may also be positive for
> NeuralTextures, and will be smaller for the graphics-based FaceSwap and
> Face2Face methods. Per-manipulation results will be treated as **directional and
> descriptive**, because each manipulation has only 30 sequences in the test set.

Naming the aggregation rule now matters: without it, "largest effect" could later
mean frame AUC, video AUC, absolute NPR performance, or the difference.

⚠️ This counts **only because it is recorded now**. Finding the pattern afterwards
and reporting it as anticipated would be the exact behaviour this paper criticises.
If no such pattern appears, that is reported as the prediction failing.

## Compression — treated as an outcome, not assumed

Heavy compression may weaken the local pixel relationships NPR targets, **or**
introduce blocking artifacts that are themselves local pixel relationships and land
in the residual. Behaviour at c40 is **uncertain** and is an experimental outcome.

❌ Do not write that c40 destroys what NPR measures. Poor performance is plausible,
not guaranteed.

## Matched-recipe conditions

Identical: fixed split, batch construction and sampling procedure, augmentations
and normalisation, focal loss (γ=2, α from train class balance), AdamW `lr=3e-4`
`wd=0.05`, cosine, 15 epochs, batch 64, 128 px, fp32, checkpoint = best validation
ROC-AUC, seeds 0–4.

⚠️ Say **"identical batch construction and sampling procedure"**, not "identical
batches", unless the sampled row order is hashed per seed.

## Interpretation rule — fixed before running

| outcome | report |
|---|---|
| Aggregation and uncertainty effects appear for NPR too | Not specific to FAD. Report alongside the primary, labelled supplementary |
| They do not appear | **A finding about representation dependence**, bounding the generality claim |
| NPR performs poorly at c40 | Report plainly, with the compression caveat above |
| Per-manipulation pattern matches the prediction | Report as a confirmed prediction |
| It does not | Report as the prediction failing |

❌ No outcome changes `canonical.json` or the primary FAD conclusion.

## Fidelity checks before the ten runs

- **Transform parity.** Our implementation must match theirs numerically on random
  tensors, to float tolerance.
- **Shape and channel count** unchanged through the transform.
- **Test membership.** The run must select exactly the same 3,000 test items as the
  committed Xception dump, matched on **full paths, labels, manipulation names,
  video groups and source-target components** — not path strings alone.
- **Range check, not equality.** A retrained plain Xception through the new path
  should land inside the observed seed range **0.77203 – 0.85272**. Requiring
  0.81347 exactly contradicts our own C2 run-to-run finding, and the V8 checkpoints
  were not retained so an exact rescore is impossible.
- **Params / FLOPs / latency** via `eff_table.py`. Expect params identical to
  Xception; FLOPs slightly higher from the interpolation.

## Sanity band — a diagnostic, not a rejection rule

Plain Xception sits near 0.81 at c40. A result far outside **0.75–0.90** triggers
checks on wiring, labels, split and optimisation.

✅ **If those checks pass, the result is reported whatever it is.** Filtering by
expectation is the behaviour this paper argues against.

## Isolation

Branch. New directories (`results/predictions_npr/` for **both** arms of this
campaign, so the same-session pairs stay together and apart from V8). `config.py` additive only.
Analysis code unchanged. Before merging:

```bash
git diff main -- src/train.py src/crossed_boot.py src/agg_space.py \
                 src/cluster_boot.py src/clusters.py     # must print nothing
python -m src.reproduce                                   # must return 16/16
```

NPR needs no new training loop, so `src/train.py` should not require changes at all.

## Run guards

Dump predictions every run · distinct `--out-dir`, 5 distinct md5s · verify the
split against the committed reference JSON · no `--amp`.

## Cost

Estimated **4–6 GPU-hours for ten runs**, from the observed cost of five Xception
runs. ⚠️ **Remains an estimate until one NPR epoch is timed**, with training time,
memory use and the projected five-seed cost recorded.
