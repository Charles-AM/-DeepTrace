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

## B · Publish the FaceForensics++ component map (nearly free)

### What ships

- One CSV: each FF++ source-target sequence mapped to its component id
- `src/clusters.py`, which already exists — union-find over the FF++ pair graph
- A short README: what a component is, why it is the right resampling unit, how to use it for grouped splitting

### Why it is worth publishing

A target-video group is **not** the strongest grouping. A video can appear as the
target of one manipulation and the source of another, so grouping by target still
lets related footage cross a split. Components merge every sequence linked by a
shared source or target, which is the unit that actually breaks the dependence.

We concede the matching limitation openly: our **split** is by target-video group
while our **resampling** is by component, and component-disjoint splitting (L3) was
never run. Publishing the map is what lets others do what we did not.

### What the audit says about the need

`docs/split-policy-audit.md`: of six recent top-tier FF++ papers, four state their
partition explicitly, **one defers to a benchmark (FreqDebias, CVPR 2025) and one
states nothing at all (UCF, ICCV 2023)**. Grouping is inconsistently documented, so
a ready-made, checkable grouping is useful.

### Accomplishes

A reusable artifact rather than a claim — the one piece of this project another
researcher can pick up and use in an afternoon.

---

## Effort and order

| step | work | depends on |
|---|---|---|
| B — export the component CSV and README | ~1 hour; the code exists | nothing |
| A — draft the four-ask paragraph | ~1 hour | nothing |
| Cite B from A4 | minutes | both |

Do **B first**: it is mechanical, and having the artifact in hand makes A's
recommendations concrete rather than aspirational.

## What must not be claimed

- ❌ "Their ranking is noise." We have not measured their resolution. ✅ "Whether it is resolvable is not reported, and is answerable."
- ❌ "No deepfake paper propagates seed variance." ✅ "Among eight papers from ECCV 2020 to CVPR 2025, none do."
- ❌ "Frame-level AUC is the field norm." ✅ Quote SBI and FreqDebias and let them disagree.
- ❌ Presenting component-disjoint splitting as something we did. We did not run L3.
