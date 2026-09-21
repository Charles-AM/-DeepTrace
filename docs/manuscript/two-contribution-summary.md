# ⛔ SUPERSEDED — do not use

Replaced by **`extended-summary.md`**, which reflects the 2026-09-18 restructure
(`thesis-and-contributions.md`) and carries five corrections this version lacks.

Retained only as a record of the earlier wording. Its numbers are correct; its
**structure** is not — it presents the crop-randomised experiment as a contribution
rather than as motivation, and omits the scope qualifiers on the final paragraph.

---

# Two-contribution summary — agreed text

Settled 2026-09-18. Every figure verified against `results/canonical.json` and
`results/analysis/crossed/README.md`. **Quote it; do not paraphrase from memory.**

---

Evaluation design determines both what a deepfake detector appears to achieve and
how much evidential weight an architectural comparison can carry. We demonstrate
this using F3-Net's frequency-aware decomposition (FAD) as a controlled case study.
FAD was selected because its originating paper reports a small, specific gain of
+0.014 AUC under the same nominal FaceForensics++ c40 compression condition studied
here, although our broader experimental settings differ.

Our first experiment changed how the data were partitioned. Using the same
underlying crop corpus, the same three architectures, and matched training
procedures, we applied either crop-randomized or video-disjoint splitting and
retrained each detector. Under crop randomization, crops from the same video could
occur in both training and testing, and every architecture achieved frame-pooled
AUC near 0.99. When the partition respected target-video-group boundaries, the same
measure fell to approximately 0.80. This produced a protocol gap of approximately
18 AUC points at c40, consistent in direction and approximate magnitude across all
three architectures, compared with approximately eight points at c23.

Our second experiment held the model comparison and predictions fixed while
changing which sources of uncertainty were propagated. We compared Xception with
Xception plus FAD across five training runs on a fixed c40 split using
video-aggregated AUC. Resampling test content alone produced a 95% interval of
[−0.0224, +0.0042], while resampling training runs alone produced
[−0.0257, +0.0041]. Both conditional intervals excluded F3-Net's reported +0.014
FAD gain. When test-content and training-run variation were propagated jointly,
using source-target components as the content-resampling unit, the interval widened
to [−0.0358, +0.0180], with a point estimate of −0.0092. This interval contained
both zero and +0.014. Each conditional analysis is valid for its narrower question,
but neither alone supports an architecture-level conclusion intended to generalize
across both test content and training runs.

Our study therefore neither demonstrated a FAD advantage nor ruled out a benefit of
the published magnitude. The conclusion is not that FAD is ineffective, but that
small architectural gains cannot be interpreted responsibly without clearly
specified partitioning, estimands, dependence units, and training-run uncertainty.

---

## Provenance of every figure

| figure | source |
|---|---|
| +0.014 | F3-Net Fig. 7(a) p.12 row 2 − row 1 (`docs/f3net-ablation-verified.md`) |
| c40 = LQ | FF++ §3 p.5, quantization 40 (ledger §2, E2 closed) |
| ~0.99 / ~0.80, ~18 pts | `results/in_domain_c40_vid/README.md` §1 |
| ~8 pts at c23 | `results/in_domain_c23_vid/README.md` §1 |
| [−0.0224, +0.0042], [−0.0257, +0.0041] | `results/analysis/crossed/README.md` §1 |
| −0.0092, [−0.0358, +0.0180] | `results/canonical.json` |

## Carried deliberately

- **"nominal ... condition ... although our broader experimental settings differ"** —
  we share the compression level with F3-Net, not the subset size, input resolution,
  training budget, or necessarily the estimand (ledger §11, §22)
- **"protocol gap"**, not leakage — the L1/L2 comparison confounds leakage with
  partition difficulty. ⚠️ **This paragraph must be revisited when V2 completes**
- **"source-target components as the content-resampling unit"** — the unit, stated
  as an assumption rather than as established independence
- **"Each conditional analysis is valid for its narrower question"** — the
  conditionals are not errors; they answer narrower questions

## Owed

The methods section must state that video aggregation averages frames **in logit
space**. Under mean-probability the point estimate is −0.0060 and the exceedance
rate above +0.014 doubles from 4.1% to 8.3% (ledger §19). "Video-aggregated" alone
does not specify the estimand.
