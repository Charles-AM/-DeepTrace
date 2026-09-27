# Literature claims we can now make — direction, pending re-verification

**Written 2026-09-26.** Each claim below was checked against a source PDF the same
day (see the two audit docs for method and verbatim evidence). **Status: DIRECTION
— awaiting Charlie's independent re-check**, as was done for four of the papers in
`docs/split-policy-audit.md`. Do not submit on these until that second pass runs.

Sources: `docs/seed-variance-audit.md` (eight papers) ·
`docs/split-policy-audit.md` (seven papers) · `docs/VERIFICATION-LEDGER.md`

---

## Contribution 1 — aggregation

**C1.1 — The convention is contested, in the field's own words.**
SBI (CVPR 2022) reports **video-level** AUC "to compare with prior works" and calls
frame averaging *"typically"* done. FreqDebias (CVPR 2025) adopts **frame-level**
AUC *"following the deepfake detection studies [8, 62, 64, 65]"*. Two CVPR papers,
opposite estimands, both framed as convention.
→ strongest available motivation for C1; it is the literature's disagreement, not ours

**C1.2** DeepfakeBench (NeurIPS 2023) standardises the estimand as frame-level.
→ pair with C1.1: standardising *one* convention does not settle which

## Contribution 2 — dependence

**C2.1** Zero of six recent FF++ papers crop-randomise. Four state their partition
explicitly, one defers to a benchmark (FreqDebias), one does not state it at all
(UCF, ICCV 2023 — zero hits for `split`, `partition`, `720`, `140`).
→ C2 is a **controlled stress test**, never a correction of common practice

## Contribution 3 — training variation

**C3.1 — Zero of eight papers propagate training-run variation** into their
reported comparisons (ECCV 2020 – CVPR 2025).

**C3.2 — DF40 (NeurIPS 2024) declines on the record.** Venue checklist, verbatim:
*"Did you report error bars (e.g., with respect to the random seed after running
experiments multiple times)? [No] We use the fixed seed that is the same as
DeepfakeBench [90] to align the settings and facilitate fair comparison."*
→ a fixed shared seed removes a confound *between* methods; it does not make the
difference generalise beyond that draw. That substitution is what C3 measures.

**C3.3** DeepfakeBench's "reproducibility" means code availability and unified
protocol. Run-to-run variance is never raised (zero seed/sd/error-bar hits).

---

## Do not write

| ❌ | ✅ |
|---|---|
| "No deepfake paper propagates seed variance" | "Among eight papers from ECCV 2020 to CVPR 2025, none do" |
| "Frame-level AUC is the field norm" | quote SBI and FreqDebias and let them disagree |
| "DF40 was careless" / "fixing the seed is an error" | "DF40 states it did not report error bars, using a fixed seed matched to DeepfakeBench" |
| "The field crop-randomises" | "one of six does not state its partition; one defers" |
| generalising F3-Net's compression motivation to frequency methods as a class | "F3-Net reports its largest FAD-related advantage under low-quality compression" |

`UNSTATED` means a paper does not discuss something. It is never evidence about
what the authors actually ran.
