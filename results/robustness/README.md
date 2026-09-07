# Robustness sweep (Phase 2, pre-protocol-fix)

⚠️ **Not used in the current paper.** Produced before the protocol fix, on
crop-randomised (L1) checkpoints, and belongs to the older "why frequency features
fail" framing that ESSENCE §12 removed.

7 models × 5 perturbations × 4 severities + clean (seed 0); 4 models × JPEG × 4
severities (seeds 1–2) — ~170 evaluations. `src/robustness.py`,
`src/jpeg_robustness.py`, `src/transforms_perturb.py`, `src/transforms_jpeg.py`.

The one finding that survived into the current framing is recorded in ESSENCE §8
via `frequency_only`: it is stable across conditions because it is weak, not
because it is well-founded. Do not cite the absolute numbers here.
