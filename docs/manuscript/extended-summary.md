# Extended summary — agreed text

Settled 2026-09-18. Supersedes `two-contribution-summary.md`, which predates the
restructure in `thesis-and-contributions.md`.

⚠️ **This is the introduction narrative, not the abstract.** Too detailed for one —
see "Abstract" below.

---

Deepfake-detector comparisons depend not only on the models being compared, but also
on how predictions are aggregated, which observations are treated as independent,
and which sources of variation are propagated. Using F3-Net's frequency-aware
decomposition (FAD) as a controlled case study, we show that these choices can
change reported quantities and the conclusions they support.

Our motivation was a failure in our own pipeline. Early experiments partitioned face
crops at random, allowing crops from one video to occur in both training and testing.
Re-running the same three architectures on the same crop corpus with matched training
procedures, changing only the partitioning rule, we measured the consequence:
frame-pooled AUC near 0.99 under crop randomisation against approximately 0.80 when
the partition respected target-video-group boundaries — roughly 18 AUC points at c40
and eight at c23, consistent across all three architectures. Our targeted audit of six
FaceForensics++ studies found no evidence that any uses crop-randomised splitting, so
we present this as a controlled stress test of train–test nonindependence rather than
a characterisation of practice. Its value is scale: the protocol gap was numerically
about thirteen times the +0.014 FAD effect our case study examines.

The remaining results concern choices that are demonstrably made. First, "AUC" does
not fully specify the estimand. Frame-pooled and video-aggregated AUC are both used
in this literature, and the rule used to form a video score was not specified in the
studies we audited. In a post-hoc sensitivity analysis, moving from mean-logit to
mean-probability aggregation left our conclusion intact but shifted the point estimate
from −0.0092 to −0.0060, widened the interval by roughly 15%, and doubled the
proportion of bootstrap replicates exceeding the reference effect, from 4.1% to 8.3%.

Second, the resampling unit can reverse a verdict without altering the estimate.
Holding the frame-pooled AUC difference fixed at −0.0196 and changing only the
resampling unit, treating crops as independent gave [−0.0325, −0.0079], excluding
zero; resampling video files gave [−0.0606, +0.0167] and source-target components
[−0.0534, +0.0096], neither of which does.

Third, architecture-level inference should propagate both test-content and
training-run variation. Across five training runs on a fixed c40 split, resampling
test content alone gave [−0.0224, +0.0042] and training runs alone [−0.0257,
+0.0041] — both excluding F3-Net's +0.014. Propagated jointly, the interval widened
to [−0.0358, +0.0180] around −0.0092, containing zero and +0.014 alike. Each
conditional analysis is valid for its narrower question; neither alone supports a
conclusion about the architecture.

Our study therefore neither demonstrated a FAD advantage nor ruled out a benefit of
the published magnitude. This is not evidence that FAD is ineffective; rather, at
this evaluation scale, the evidence cannot distinguish no advantage from a gain of
+0.014. Small architectural effects cannot be interpreted responsibly without
specifying the data partition, estimand, dependence unit, and sources of variation
propagated in the uncertainty analysis.

---

## Five phrasings that are load-bearing

| ✅ written | ❌ not |
|---|---|
| "not specified in the studies we audited" | "typically unstated" — six papers cannot establish what a literature typically does |
| "In a post-hoc sensitivity analysis" | presenting the aggregation comparison as planned (ledger §19.3) |
| "proportion of bootstrap replicates exceeding the reference effect" | "probability that FAD works" — a bootstrap tail proportion is not that |
| "architecture-level inference **should** propagate" | "**requires**" — one comparison on one subset does not establish universality |
| "the evidence cannot distinguish no advantage from a gain of +0.014" | "not because FAD is ineffective, but because…" — **claims to know why we failed to distinguish. We do not. FAD's true effect could be zero.** |

The last is the one that survived three drafts before being caught.

## Abstract

This text is the **introduction narrative**. A conventional abstract should retain:

- the **18-point** stress-test result
- the **−0.0196** resampling example (one estimate, verdict reversed)
- the final **crossed interval [−0.0358, +0.0180]**

and omit the secondary numbers — the c23 comparison, the aggregation-space shifts,
and the individual conditional intervals.

## Provenance

Every figure traces to `results/canonical.json`,
`results/analysis/crossed/README.md`, `results/analysis/cluster_boot/ALL.csv`,
`results/in_domain_c40_vid/README.md`, `results/in_domain_c23_vid/README.md`, or
ledger §19. Verified 2026-09-17 and 2026-09-18.
