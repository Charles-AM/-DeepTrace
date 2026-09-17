# Contribution map

How the three contributions fit together, what each rests on, and where its
evidence lives. Read this before drafting any section; read the per-contribution
evidence maps before writing any sentence.

---

## The line the paper is built around

> **The conclusion was stable; its warrant was not.**

## The fact that unifies all three

> **3,000 test crops. 150 video files. 30 target groups. 29 components.**

Apparent precision came from frame count; real resolution comes from independent
clusters. Every contribution follows from that gap.

Verified from the raw dumps, not the READMEs: `video_id == target_seq` in 3000/3000
rows; all 30 groups carry the real video and all four manipulations; 28 of 30
targets have their manipulation partner in training; exactly one pair (251, 375)
has both members in test, which is what collapses 30 groups into 29 components
(ledger §15).

---

## The decision chain

A researcher comparing two detectors on FF++ makes five choices. Each has a
measured cost. **Each contribution owns part of the chain.**

| step | the choice | measured cost | owner |
|---|---|---|---|
| 1 | **How do I split?** crop-randomised or video-disjoint | **~18 AUC points** at c40 | **C1** |
| 2 | **What do I compute?** frame-pooled or video-aggregated | moves the estimate in **5 of 5** runs | **C1** |
| 3 | **In what score space?** mean-logit or mean-probability | exceedance rate **4.1% → 8.3%** | **C1** |
| 4 | **What unit carries the uncertainty?** frame, video, component | **verdict flips** — frame excludes zero, clustered units do not | **C2** |
| 5 | **Do I propagate training-run variation too?** | crossed interval **1.8× wider**; both conditionals exclude +0.014, the crossed one does not | **C2** |
| — | **What may I then conclude about the architecture?** | neither demonstrated nor excluded | **C3** |

Steps 1–3 determine **which quantity is estimated**. Steps 4–5 determine **how
uncertain that estimate is**. Step 6 is what happens when a real published claim is
run through the corrected chain.

---

## The three contributions

### Contribution 1 — protocol and estimand

> Protocol and estimand choices change apparent performance and the apparent size
> of architectural contrasts.

**Owns:** steps 1–3. **Anchor:** Kapoor & Narayanan **[L3.2] Nonindependence
between training and test samples** — a **splitting** error.
**Evidence:** `docs/contribution-1-evidence.md`.

Headline numbers: ~18-point protocol gap at c40, consistent across three
architectures (17.6 / 18.2 / 19.2); ~8 points at c23, a difference-in-differences of
+9.5 to +10.4; L1 compresses the three architectures into 0.6 points of separation
against 2.1 under L2.

**Status:** dissected, evidence-mapped, wording settled (2026-09-17).

### Contribution 2 — dependence-aware inference

> Dependence-aware evaluation changes test-content uncertainty and the statistical
> verdicts drawn from it. Architecture-level conclusions additionally require
> propagating training-run variation.

**Owns:** steps 4–5. **Anchor:** Hurlbert, **pseudoreplication** — an **inference**
error. **Evidence:** `docs/contribution-2-evidence.md`.

Headline numbers: one estimate (−0.0196) under three uncertainty models gives two
conclusions; component half-widths 0.037–0.062 = 2.6–4.5× the reference; the crossed
interval [−0.0358, +0.0180] is 1.8× wider than either conditional, and **only the
crossed one fails to exclude +0.014**; non-additivity ratio 1.697.

**Status:** positioning settled, all numbers independently validated, evidence map
written (2026-09-17). **A1 still owed** — it retires limitation 4.

### Contribution 3 — the FAD case study

> Applying both corrections to a published architectural claim.

**Owns:** the conclusion. **Reference effect:** F3-Net's own FAD ablation, **+0.014
AUC**, measured on the LQ task — which FF++ defines as H.264 quantization 40, i.e.
**the same c40 setting as ours** (ledger §2, E2 closed).

Result, frozen in `results/canonical.json`: **−0.0092, 95% CI [−0.0358, +0.0180]**,
containing zero, +0.014 (F3-Net) and +0.0010 (DeepfakeBench). FAD costs ≈ **+3%**
compute.

**Status:** dissected in conversation, result frozen and reproduced exactly.
**No evidence map yet** — the remaining gap.

---

## Why these are one paper and not three

Each contribution is the precondition for the next.

**C1 makes C2 possible.** Under crop-randomised splitting every architecture scores
~0.99 and the three sit 0.6 points apart against the ceiling. There is no room for
an uncertainty analysis to say anything. Fixing the protocol restored headroom.

**C1 and C2 together make C3 meaningful.** A −0.0092 point estimate means nothing
without knowing that the design's resolution is 2.6–4.5× the effect being tested.
The case study's value is not its point estimate but its interval.

**C3 supplies the stakes for C1 and C2.** Without a real published claim to run
through the chain, the methodology is a demonstration in search of an application.
With it, each measured cost has a referent: 18 points against 1.4, a verdict that
flips on the choice of unit, a conditional interval that would have excluded the
published effect.

---

## The ladder — every choice dwarfs the effect

| quantity | AUC points | owner |
|---|---|---|
| Protocol choice (crop vs video split) | **~18** | C1 |
| Compression's effect on that gap | **~10** | C1 |
| Test-content uncertainty (component half-widths) | 3.7 – 6.2 | C2 |
| Run-to-run variation, split frozen | 3.4 | C2 |
| Repeatability movement, nominally identical config | 3.4 | C2 |
| Crossed interval half-width | 2.7 | C2 |
| **The published effect under debate (+0.014)** | **1.4** | C3 |
| Aggregation-space shift in the point estimate | 0.3 | C1 |
| Our point estimate for FAD | 0.9 | C3 |

**Every source of variation we measured is larger than the effect the literature
argues about.** That ordering is the paper.

---

## Two errors, two anchors — do not cross them

| error | what it is | anchor | contribution |
|---|---|---|---|
| **Nonindependence between train and test** | crops from one video on both sides of the split | Kapoor & Narayanan, *Patterns* 4(9), 100804, 2023 (PMID **37720327**), [L3.2] | **1** |
| **Pseudoreplication** | treating correlated crops as independent *inferential* units | Hurlbert, *Ecol. Monogr.* 54(2), 187–211, 1984 | **2** |

A study can commit either without the other. Citing Hurlbert for the splitting
problem, or Kapoor & Narayanan for the uncertainty problem, misattributes both.

---

## Where we sit relative to the literature

| work | what they established | what we add |
|---|---|---|
| **F3-Net** (ECCV 2020) | a frequency branch improves detection under heavy compression; +0.014 for FAD alone | what resolution an evaluation of that design can achieve |
| **DeepfakeBench** (NeurIPS 2023) | standardises **which** estimate is computed, for comparability | **how uncertain** that estimate is |
| **Bouthillier et al.** (MLSys 2021) | training-run variance in general ML, under an explicit i.i.d. assumption | the **clustered** case they flag in one sentence and set aside |
| **Kapoor & Narayanan** (2023) | leakage taxonomy across 17 fields; correcting it dissolved an apparent ML advantage in civil-war prediction | the deepfake-detection instance, with the cost measured |

⚠️ No adversarial framing is needed anywhere in that table. Describe what the work
did, state its boundary, claim only the remainder. The precision is what makes the
criticism unnecessary.

---

## Status

| | C1 | C2 | C3 |
|---|---|---|---|
| dissected | ✅ | ✅ | ✅ (conversation) |
| evidence map | ✅ | ✅ | ❌ **owed** |
| numbers independently validated | ✅ §15 | ✅ §17–19 | ✅ §18.1 exact |
| wording settled | ✅ | ✅ | partly |
| experiments owed | B2 (V2) | **A1**, A4, A5, A6, C2 | — |

**Shared documentation debt:** the estimand sentence in `canonical.json` (blocked on
approval), the §11 limitations sentence, a replacement intro paragraph, a results
directory for the aggregation sensitivity, and 7 unverified citations.

---

## How to use this map

1. **Before drafting a section** — read this file to place it in the chain.
2. **Before writing a sentence** — read that contribution's evidence map. If the
   sentence is not derivable from a row there, it does not go in.
3. **Before citing a paper** — check `docs/VERIFICATION-LEDGER.md`. 19 sections,
   every external claim traced to a quote.
4. **When tempted by a strong phrasing** — check the do-not-write tables. Between
   them they hold 28 entries, most of which took a round of review to settle, and
   several of which were reintroduced by accident within hours of being written down.
