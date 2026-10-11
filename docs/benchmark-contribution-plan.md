# Plan — turning our results into benchmark recommendations

**Written 2026-10-11.** Two low-cost additions that convert this study from "we
measured something" into "here is what benchmarks should change." Both draw only on
results already committed and verified; neither needs a new experiment.

Rejected for now: re-running another benchmark's checkpoints to compute its
resolution floor, and maintaining a pull request against DeepfakeBench. Both are
real work with no benefit to the course project. They are the follow-up.

---

## A · A recommendations paragraph (free)

Four asks. **Each one is a result we already have, restated as a protocol change.**

### A1 — State which estimand the number is

| | |
|---|---|
| **Evidence** | In V2, the same comparison gives **+0.1998** frame-pooled and **+0.1731** video mean-logit. The gap is **0.0267 — 1.9× the +0.014 reference effect.** In the main c40 comparison the estimand moves the point estimate by **0.00286**, about 20% of the reference. |
| **Supporting** | SBI (CVPR 2022) reports video-level AUC and calls frame averaging "typical"; FreqDebias (CVPR 2025) adopts frame-level AUC "following" four studies. Two CVPR papers, opposite estimands, both described as convention (`docs/seed-variance-audit.md`). |
| **Accomplishes** | Turns contribution 1 from a measurement into an actionable ask, and the supporting evidence comes from the literature rather than from us. |

### A2 — Report more than one training run

| | |
|---|---|
| **Evidence** | The strongest single fact in the project. Both conditional analyses **exclude** +0.014 (half-widths 0.0133 and 0.0149); the crossed analysis **includes** it (0.0269). **Which variation you propagate flips the verdict**, on identical data and identical models. |
| **Supporting** | 0 of 8 recent papers propagate training-run variation. DF40 (NeurIPS 2024) is asked directly by the venue checklist and answers **[No]**. |
| **Accomplishes** | Contribution 3 becomes a protocol ask backed by the field's own on-record admission. |

### A3 — Use cluster-aware intervals, and publish a resolution floor

| | |
|---|---|
| **Evidence** | Our test split holds 150 videos but only **29 source-target components**. Single-seed component-bootstrap half-widths at c40 are **0.037–0.062 AUC**, 2.6–4.5× the reference effect. |
| **The ask** | A benchmark that ranks methods should report the smallest difference its protocol can resolve, and mark leaderboard gaps below it as unresolved. DeepfakeBench ranks F3Net (0.8271) above Xception (0.8261) — a **0.0010** gap — with no statement of whether that is resolvable. |
| **Accomplishes** | Generalises contribution 2's machinery into a practice any benchmark can adopt. |

⚠️ **Boundary.** Our half-widths come from 150 pairs; DeepfakeBench uses full FF++,
so **its resolution is better than ours and our numbers must not be transferred to
it**. The contribution is the question and the method, never a verdict on their
ranking. Write *"whether a 0.0010 difference is resolvable under that protocol is
not reported, and is answerable"* — never *"their ranking is noise."*

### A4 — Store per-sample predictions

| | |
|---|---|
| **Evidence** | Every uncertainty analysis in this project runs on CPU over stored prediction files; none needs retraining. 34 dumps, ~650 KB each. |
| **First-hand cost of not doing it** | Our own **C1 and C2 runs have no dumps** — `train.py` wrote checkpoints but not per-frame scores — so both are aggregate-only and recoverable only while the Kaggle outputs survive (`docs/ARTIFACT-POLICY.md`). We recommend this because it cost us. |
| **Feasibility for others** | DeepfakeBench's `training/test.py` already builds `prediction_lists` and `label_lists` per sample and discards them. Writing them out is a few lines. |
| **Accomplishes** | Makes A1–A3 nearly free for anyone else: all three are re-analyses of stored predictions, not new training. |

---

## B · Component map — ⛔ WITHDRAWN 2026-10-11, the premise was wrong

**Do not do this.** Proposed on the assumption that source-target components are
non-trivial clusters worth publishing. They are not. Checked the same day against
the official FF++ split files (`dataset/splits/{train,val,test}.json`, from
ondyari/FaceForensics):

| | |
|---|---|
| pairs | **500** |
| unique sequences | **1,000** |
| sequences appearing in more than one pair | **0** |
| components | **500, every one of size exactly 2** |
| components shared across the official train/val/test splits | **0** |

Union-find has nothing to merge. **The component map is the pair list**, and the
official 360/70/70 pair split is already component-disjoint. Publishing a CSV that
restates the dataset's own pairing adds nothing.

### What the check produced instead — keep this

> The official FaceForensics++ split is already component-disjoint. The dependence
> risk does not come from the dataset's design; it comes from subsampling or
> re-splitting it.

Sharper than the "grouping is inconsistently documented" line this file originally
carried, and it reframes the split audit: **UCF (ICCV 2023) not stating its
partition matters precisely because departing from the official split is where the
risk enters.** Fold this into A3 as a one-line sharpening.

Two side findings:

- **Independent confirmation of ledger §7.** 360 pairs × 2 = **720** training
  videos, 70 × 2 = **140** each for val and test — the 720/140/140 already recorded.
- **Our own 30 target groups → 29 components is explained**: exactly one of our 30
  test targets is paired with another of them. One merged pair, not a systematic gap.

### The residual mechanism, unquantified

A sequence can be a *target* in train while being the *source* of a test video.
Target-grouping does not prevent that; component-grouping does. How often it occurs
in our split is **unknown** — it needs the V2/L2 manifests, which are gap 1 in
`docs/recovery-runbook.md`. State it as a mechanism, never as a measured rate.

## Effort and order

| step | work | depends on |
|---|---|---|
| A — draft the four-ask paragraph | ~1 hour | nothing |

B is withdrawn, so A is the whole of this plan. **Neither is in scope now** — the
programme proceeds on the three original contributions; this file is a parked
think-piece, kept because the component finding above is worth reusing.

## What must not be claimed

- ❌ "Their ranking is noise." We have not measured their resolution. ✅ "Whether it is resolvable is not reported, and is answerable."
- ❌ "No deepfake paper propagates seed variance." ✅ "Among eight papers from ECCV 2020 to CVPR 2025, none do."
- ❌ "Frame-level AUC is the field norm." ✅ Quote SBI and FreqDebias and let them disagree.
- ❌ Presenting component-disjoint splitting as something we did. We did not run L3.
- ❌ "Grouping is inconsistently documented, so a component map is useful." Withdrawn — see B.
