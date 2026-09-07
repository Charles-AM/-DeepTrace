# Prespecification — target-group vs component resampling

**Written and committed before the analysis exists.** No target-group interval has
been computed at the time of this commit; `cluster_boot` and `crossed_boot`
currently offer frame / video / component only.

## The claim being tested

`results/analysis/resolution/README.md` and the recommendations draft state that
source-target components merged only one additional pair of target groups (29 vs
30), and an earlier draft inferred from that count that components "added almost
nothing". **Similar unit counts do not guarantee similar uncertainty**, so the
inference is unsupported. This measures it.

## What will be run

Adding `target` (grouping on `target_seq` alone) as a fourth resampling unit.

| # | analysis | comparisons | replicates | RNG seed |
|---|---|---|---|---|
| A | conditional per-seed intervals | the 8 already reported in `../results/analysis/thresholds/` — c40 seeds 0–4, c23 seeds 0–2, Xception vs Xception+FAD, video-level | 2,000 | 0 |
| B | **the crossed interval** — the primary result | V8 fixed split, 5 runs, video-level | 50,000 | 0 |

B matters more than A: if the headline interval depends on the unit choice, that
bears directly on the paper's primary result rather than on a side remark.

## Primary metric — fixed in advance

1. **Absolute difference in half-width**, target minus component, per comparison.
2. **Ratio** of half-widths, since 0.001 on a 0.02 half-width is 5%.
3. **Whether any exclusion verdict at the +0.014 reference changes.**

For B additionally: the exceedance rate above +0.014 under each unit, and whether
the prespecified case-1 verdict in `crossed-prespecification.md` still holds.

## Interpretation — fixed in advance

**Component resampling remains the preferred inferential unit regardless of
outcome**, because it captures dependence through the source-target graph that
target grouping cannot see. This analysis measures how much that choice cost or
saved in this split; it is not a search for a cheaper unit.

| outcome | conclusion |
|---|---|
| half-widths near-identical, no verdict changes | *Target-group resampling approximated component-level uncertainty in this split.* |
| half-widths differ materially, or any verdict changes | the single merge matters more than the count suggests — report the difference |

⚠️ **Scope, whatever the result.** This is one fixed split of one FF++ subset. It
cannot establish that target grouping is generally sufficient for FF++, and the
recommendation stays: *identify the relevant dependence structure empirically
rather than assuming either crops or the strictest available grouping is
automatically appropriate.*

## Tagging

`results-frozen-2026-09-06` is **immutable** and will not be moved
(`results/PROVENANCE.md`). Results from this analysis get a new tag.
