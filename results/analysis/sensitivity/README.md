# Class-stratified bootstrap sensitivity

Every reported interval uses the standard **unstratified** cluster bootstrap. This
is the sensitivity check: the stratified variant holds each replicate's class
composition fixed.

**Conclusions were unchanged.** 0 of 10 exclusion verdicts flipped; video-unit
half-widths moved by at most 0.0011 (2.4%).

At the **component unit the two procedures are bit-identical** in all five seeds.
That is structural, not luck: a component carries both real and fake videos, so
every cluster is mixed, there is one stratum, and stratification is a no-op. It
can only matter for the video unit, where a file is entirely one class.

`stratified_bootstrap.csv` — 20 rows (5 seeds × 2 units × 2 procedures).
Produced via `src.cluster_boot.paired_bootstrap(..., stratified=True)`.
