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

## Scope

Five seeds of **Xception + NPR** on the **fixed c40 video-disjoint split**
(`split_seed 0`), dumps saved, paired against the five Xception seeds already
committed there.

| experiment | covered |
|---|---|
| 3 — aggregation | ✅ |
| 4 — units, conditional vs crossed | ✅ |
| 1 — splitting | ❌ second campaign |
| 2 — prior exposure | ❌ manifests uncommitted |

## 🚨 Directional prediction — written before running

NPR targets artifacts from **learned upsampling**. The four FF++ methods differ:

| method | mechanism | NPR expected to help |
|---|---|---|
| Deepfakes | two autoencoders, shared encoder | **most** |
| NeuralTextures | learned neural texture + rendering network | plausibly |
| FaceSwap | classical graphics: landmarks, 3D template, blending | least |
| Face2Face | 3D reconstruction + image-based rendering | least |

**Prediction: the largest NPR effect appears on Deepfakes, plausibly on
NeuralTextures, and least on FaceSwap and Face2Face.**

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

## Fidelity checks before the five runs

- **Transform parity.** Our implementation must match theirs numerically on random
  tensors, to float tolerance.
- **Shape and channel count** unchanged through the transform.
- **Path-level identity.** The run must select exactly the same 3,000 test items as
  the committed Xception dump — compare path strings.
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

Branch. New directories (`results/predictions_npr/`). `config.py` additive only.
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

Estimated 2–3 GPU-hours for five runs, from the observed cost of five Xception
runs. ⚠️ **Remains an estimate until one NPR epoch is timed.**
