# Descriptive checks — answers to an external critique (2026-10-02)

⚠️ **NOT PRESPECIFIED. DESCRIPTIVE ONLY.** Neither result below may be reported as
a confirmatory finding. Reproduce with `python -m src.descriptive_checks`
(CPU, seconds, no training).

An external critique raised seven design points. Four were already satisfied, one
was a genuine gap (now fixed, see below), and two were factual questions about the
data answerable from committed dumps. Those two are recorded here.

---

## 1. Frames per video sequence — the concern does not apply

**Claim examined:** *"unequal frame counts change the weighting and stability of
video scores."*

**Result: every dump has EXACTLY 20 frames per sequence, 150 sequences.** All ten
fixed-split c40 dumps, `min = max = 20`. Extraction used
`--every 12 --max-per-video 20`, and every retained sequence reached the cap.

→ Video-level scores are **equally weighted by construction**. This is a fact to
state, not a limitation to disclose.

## 2. Per-manipulation AUC — the pooled null is not hiding anything

**Claim examined:** *"a pooled result can conceal a gain driven by one easier
manipulation method."*

Video-aggregated AUC difference (Xception+FAD − Xception), mean over 5 training
seeds, fixed c40 split. Each manipulation scored against all 30 real sequences.

| manipulation | mean diff | min | max |
|---|---|---|---|
| Deepfakes | **−0.0116** | −0.0478 | +0.0233 |
| Face2Face | **−0.0078** | −0.0511 | +0.0211 |
| FaceSwap | **−0.0104** | −0.0344 | +0.0189 |
| NeuralTextures | **−0.0069** | −0.0556 | +0.0167 |
| **POOLED** | **−0.0092** | −0.0392 | +0.0119 |

**All four are negative.** Spread across manipulations = **0.0047**, well under the
+0.014 reference. No manipulation type carries a hidden FAD advantage.

✅ **Cross-check: the pooled value reproduces `results/canonical.json` exactly
(−0.0092).** That is what validates the script — it runs the canonical aggregation
(mean `logit_margin` per video via `src.cluster_boot._video_key`, AUC from
`src.resolution_curve.auc`) on the canonical dumps.

**Worth noting for contribution 1:** manipulation type moves the answer by 0.0047.
The *estimand* choice moves it by 0.0267 — about 5.7× more. Which forgery type you
look at matters less than how you aggregate the scores.

---

## Two defects found while writing this — recorded because they were silent

1. **Wrong dumps.** The first run used `results/predictions/ffpp_c40_vid_*`, which
   are the **varying-split** runs, not the fixed-split runs canonical uses
   (`results/predictions_v8/`, which carry a `split_seed` column). It produced a
   pooled difference of **+0.0032** against canonical's −0.0092 — opposite sign.
   Caught only because the script prints the cross-check. **Keep that cross-check.**
2. **Meaningless pooled frame count.** The summary line unioned video keys across
   dumps, so shared sequences accumulated counts (min 40, max 360) and it concluded
   "frame counts VARY" — the opposite of the truth. Counts are only meaningful
   within a dump.

Both are the same defect class the project keeps catching: output that looks like
an answer while describing the wrong thing.

---

## The one real gap the critique found

**F3-Net trained with cross-entropy; we use focal loss.** Our comparability table in
`docs/f3net-ablation-verified.md` listed six protocol differences and omitted the
loss. A seventh row is now there, with F3-Net's two verbatim statements.

Not fixed by retraining, and deliberately so: focal loss applies identically to both
arms and cannot produce a difference between them, the programme is prespecified and
complete, and changing the recipe in response to commentary on finished results is
the exact behaviour this paper argues against.

## What the critique got wrong — its own sources

Six claims (FF++'s 720/140/140 split, F3-Net's architecture, the 0.893/0.907
ablation, F3-Net's loss, the cluster definition, per-manipulation reporting) all
cited **arXiv:2203.02115**, which is *"Towards Benchmarking and Evaluating Deepfake
Detection"* (Lin et al.) — neither FF++ nor F3-Net. Grep of that paper:
`0.893` 0 hits · `0.907` 0 hits · `720` 0 hits · `140` 0 hits ·
`cross-entropy` 0 hits · `MixBlock` 0 hits.

The underlying *facts* were mostly right and independently confirmed from the real
papers. The citations were not. ⚠️ **Do not propagate arXiv:2203.02115 into our
reference list for any of these claims.**

Its two PMC citations were correct: PMC10499856 is Kapoor & Narayanan
(PMID 37720327 ✓), PMC10728486 is a genuine correlated-AUC bootstrap paper
(*Frontiers in Veterinary Science*, 2023).
