# Proposal supplementary — settled wording, with what was rejected

**2026-10-02.** Framing and vocabulary decisions for the proposal, arising from an
external review. Records the **approved** sentences, the **rejected** ones, and why.
The proposal prose itself stays off-repo; this file holds only the decisions and
their verification.

Companions: `docs/manuscript/literature-claims.md` ·
`docs/VERIFICATION-LEDGER.md` · `docs/seed-variance-audit.md`

---

## Research question (approved)

> Under controlled training conditions, how do data-splitting strategy, score
> aggregation method, and variation across training runs affect estimated AUC
> differences and uncertainty when comparing Xception with an Xception+FAD deepfake
> detector on FaceForensics++?

## Novelty statement (approved, after three corrections)

> This project proposes a controlled evaluation study of how experimental design
> choices affect conclusions in deepfake detection. Rather than introducing a new
> detector, it will compare a standard Xception detector with an Xception model
> augmented by F3-Net's frequency-aware decomposition component. The study will test
> whether reported performance differences depend on data-splitting strategy,
> frame-to-video score aggregation, and the treatment of variation across training
> runs. **Uncertainty will be estimated at three nested levels — crops, video files,
> and source-target components — so that the effect of the assumed unit of
> independence is measured rather than assumed.**

## Prior work (approved)

> Existing work has established that deepfake-detector comparisons can be distorted
> by inconsistent preprocessing, datasets, and evaluation protocols. DeepfakeBench,
> for example, advocates standardized pipelines and evaluation procedures for fairer
> comparisons. F3-Net reports that its frequency-aware decomposition component
> improves Xception's reported AUC on FaceForensics++, but its ablation result is
> reported as a point estimate rather than as a repeated-run, cluster-aware
> uncertainty analysis. Kapoor and Narayanan further motivate the proposed
> video-disjoint design by identifying nonindependence between training and test
> samples as a form of leakage.

✅ **The F3-Net point-estimate claim is verified.** Grep of arXiv:2007.09355v2:
`seed` 0 · `±` 0 · `error bar` 0 · `standard deviation` 0 · `repeated` 0. The one
`std` hit is an email domain in the author block; the four `variance` hits are
hand-crafted *noise variance* features. Consistent with `docs/seed-variance-audit.md`.

## The gap — two versions, and they are not interchangeable

**Proposal** (hedged, because eight papers is what we have read):

> To our knowledge, previous work has not jointly isolated the effects of
> within-video nonindependence, frame-to-video aggregation, and training-run
> stochasticity on the statistical conclusion of a controlled deepfake-detector
> ablation.

**Paper** (specific, because the audit supports more):

> Among eight FF++ deepfake-detection papers from ECCV 2020 to CVPR 2025, none
> propagate training-run variation into their reported comparisons. DF40
> (NeurIPS 2024) states that it did not report error bars with respect to the random
> seed, using a fixed seed matched to DeepfakeBench.

Do not use the paper version in the proposal: it reports an audit result, and the
proposal is written before results.

## What this project is **not** claiming

1. That FAD is a new method.
2. That Xception+FAD will necessarily outperform Xception.
3. That crop-randomised evaluation is a valid deployment estimate.
4. That a non-significant result proves no detector improvement exists.

Item 4 matches the canonical verdict exactly: *neither demonstrated a FAD advantage
nor excluded a gain of the published magnitude.*

---

## Rejected wording — and why

| ❌ rejected | reason | ✅ use |
|---|---|---|
| "video-family-level resampling" | A coined term absent from our code, ledger and `reproduce.py`. Worse, it implies family-level **splitting**, which we never did — L3 component-disjoint was planned and not run. Our split is by target video (30 groups); resampling is by component (29) | "source-target components" |
| "practically meaningful" | We assess no deployment cost or utility threshold. The claim is statistical resolvability against an externally fixed reference | "resolvable" |
| resampling described at one level | The contribution is the **contrast** across crop / video / component, not a choice of the coarsest unit | the three-level sentence above |
| arXiv:2203.02115 as a source | *"Towards Benchmarking and Evaluating Deepfake Detection"* (Lin et al.) — contains none of the claims attributed to it (`0.893`, `0.907`, `720`, `140`, `cross-entropy`, `MixBlock`: all 0 hits). See `docs/descriptive-checks.md` | the real papers |

## Standing precision rules this file does not supersede

- Never "DeepfakeBench identifies frame pooling as a fairness problem" — pooling is
  their **solution**.
- Never "frame-level AUC is the field norm" — unsurveyed. Quote SBI and FreqDebias
  and let them disagree.
- Never present our training as replicating F3-Net's: they used cross-entropy, we use
  focal loss (`docs/f3net-ablation-verified.md`, seventh protocol row).
- Never report `f3net` / `xception_fad` as "F3-Net" — it is the FAD branch alone.
