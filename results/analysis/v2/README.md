# V2 — the seen-video advantage

**The experiment the "protocol gap" hedge was waiting on.**

## Design

One model. Trained once on the seen manifest. Scored against two test sets matched
on video count, crop count, group structure, manipulation mix, compression and
sampling position within each clip. **The only difference is whether the model
trained on those videos.**

Architecture, weights, training data and checkpoint-selection rule are all held
constant.

⚠️ **Composition is matched; intrinsic difficulty is not controlled.** The two sets
contain **different video groups**. Assignment was random under the seeded L2 split,
so no systematic bias is expected — but with 30 groups per side the realised
difficulty may differ. An earlier version of this file said partition difficulty was
"removed by construction". **That overstated it.**

## Result

| | AUC |
|---|---|
| seen (crops withheld from videos the model trained on) | **0.9884** |
| unseen (crops from videos never seen) | **0.7886** |
| **seen-video advantage** | **+0.1998** |

Reported under **three estimands**, because contribution 1 is that estimand choice
changes reported quantities — reporting one would undercut it:

| estimand | see |
|---|---|
| frame-pooled (primary) | `v2_advantage.json` |
| video-aggregated, mean-logit | same |
| video-aggregated, mean-probability | same |

Intervals use **independent component resampling** within each set.

## In context

| | |
|---|---|
| L1 crop-randomised | 0.9933 |
| **V2 seen** | **0.9884** |
| **V2 unseen** | **0.7886** |
| L2 video-disjoint | 0.8169 |
| protocol gap (L1 − L2) | ~0.18 |

The seen condition lands essentially where crop-randomised evaluation does, and the
advantage is of the same magnitude as the entire protocol gap — **under matched
composition**, which the protocol gap is not.

## ⚠️ Scope — three limits, all binding

1. **Conditional on one trained model.** Test-content variation only; **no
   training-run variation propagated.** A mechanism experiment, not a
   population-level protocol estimate.
2. **A different quantity from the 18-point gap.** That compares two models on two
   different partitions; this compares one model on matched sets. ✅ *"the
   seen-video advantage is +0.1998"* ❌ *"X of the 18 points is leakage"*.
3. **Independent, not paired, resampling.** The two sets contain **disjoint video
   groups**, so the difference is not a paired statistic. `seen_unseen_boot.py`
   refuses to run if they share components.

## What it licenses

The ~18-point protocol gap is **not plausibly explained by test-partition
difficulty alone**: same-video overlap reproduces an effect of comparable magnitude
under matched composition. The word "leakage" becomes usable for the mechanism —
though "protocol gap" remains correct for the L1/L2 contrast itself, which still
confounds two things.

## Regenerate

```
python -m src.verify_v2 --seen ... --unseen ... --manifest-seen ... --manifest-unseen ...
python -m src.seen_unseen_boot --seen ... --unseen ... --n-boot 50000 --boot-seed 0
```

Both read only the committed dumps in `results/predictions_v2/`. No GPU.
