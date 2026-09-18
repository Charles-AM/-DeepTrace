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

## Result — computed 2026-09-18, 50,000 replicates, boot seed 0

| estimand | seen | unseen | advantage | 95% CI | half-width |
|---|---|---|---|---|---|
| **frame-pooled** (primary) | 0.9884 | 0.7886 | **+0.1998** | **[+0.1422, +0.2542]** | 0.0560 |
| video-aggregated, mean-logit | 0.9992 | 0.8261 | +0.1731 | [+0.1044, +0.2350] | 0.0653 |
| video-aggregated, mean-probability | 0.9992 | 0.8214 | +0.1778 | [+0.1106, +0.2383] | 0.0639 |

**All three exclude zero decisively.** Lower bounds are +0.10 to +0.14 — an order of
magnitude above the +0.014 reference effect.

### The estimand spread is itself a finding

The advantage ranges **+0.1731 to +0.1998** depending only on how predictions are
aggregated — a spread of **0.0267**, which is **1.9× the +0.014 architectural effect
the paper's case study examines.**

Contribution 1 demonstrating itself inside contribution 1's own experiment: even
when measuring leakage, the estimand choice moves the answer by nearly twice the
effect under debate. This is why the result is reported under all three rather than
one.

### Resolution — the same apparatus, two very different outcomes

Same 29-ish components, same bootstrap, same codebase:

| comparison | effect | half-width | effect ÷ half-width | verdict |
|---|---|---|---|---|
| **V2 seen-video advantage** | 0.1998 | 0.0560 | **3.6×** | **excludes zero** |
| FAD crossed comparison | 0.0092 | 0.0269 | **0.34×** | cannot resolve |

**The evaluation is not underpowered in general.** It resolves a 20-point effect
decisively and cannot resolve a 1.4-point one. That is the resolution argument made
with measurements rather than assertion, and it retires the objection that the FAD
null is merely an artefact of a weak method.

⚠️ This is a **post-hoc observation**, not a prespecified power analysis. It is a
calibration statement about this apparatus on these components — not a general
claim about detectable effect sizes.

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

---

## Concordance with the protocol gap — match the estimands

| | estimand | value |
|---|---|---|
| Xception protocol gap (L1 − L2), c40 | **frame-pooled** | **+0.1764** |
| three-architecture range | frame-pooled | +0.176 to +0.192 |
| **V2 seen-video advantage** | **frame-pooled** | **+0.1998** |

✅ **Compare frame-pooled to frame-pooled.** Both figures above are frame-pooled, so
this is like-for-like.

⚠️ **The protocol gap cannot be computed under video aggregation**, because the L1
campaign has **no prediction dumps** (`contribution-1-evidence.md` §0). Frame-pooled
is the only estimand on which the two can be compared at all.

So for the concordance statement, V2's **frame-pooled** figure is the one to quote —
even though `results/canonical.json` uses **video-aggregated mean-logit** for the FAD
comparison. That difference is deliberate and should be stated: each comparison is
made on the estimand its counterpart supports.

❌ Do not set V2's video mean-logit (+0.1731) beside the protocol gap (+0.1764).
Numerically close, different estimands, and the closeness is coincidental.

## Agreed manuscript wording

> In a controlled mechanism experiment, we evaluated one trained Xception model on
> balanced sets of withheld crops from represented and entirely unseen video groups.
> Exact evaluation crops were absent from training in both conditions. Under
> video-level mean-logit aggregation, AUC was 0.9992 on represented groups and
> 0.8261 on unseen groups, an observed advantage of +0.1731 with a
> source–target-component bootstrap interval of [+0.1044, +0.2350]. Frame-pooled and
> mean-probability analyses produced advantages of +0.1998 and +0.1778
> respectively, with all intervals excluding zero.

Followed immediately by the scope:

> This interval captures test-content uncertainty conditional on one trained
> Xception model; it does not propagate training-run variation or establish that
> crop-randomised splitting is common in published FF++ studies.

**"Controlled verification", not "replication".** Different experimental designs
producing concordant magnitudes.

## ⛔ NOT YET FREEZABLE

The two prediction CSVs are **not committed**. Until they are, this result is
**aggregate-only** — the same state as L1, C1 and C2 — and `python -m src.reproduce`
prints `SKIP` for it.

Freezing a result whose substrate exists only in a Kaggle session is the failure
this repository documents in three other places. **Commit the dumps first.**
