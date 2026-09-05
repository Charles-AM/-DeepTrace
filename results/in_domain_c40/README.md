# c40 in-domain results — C7, the pivotal experiment (2026-09-05)

FaceForensics++ **c40** (heavy compression), 3 configs × 3 seeds, 15 epochs.
Same 150 video pairs as the c23 matrix (`--num-pairs 150`), same seeded split
procedure, so this is a **paired** c23↔c40 comparison. Environment snapshot from
the same container that produced these numbers: `docs/pip_freeze_2026-09-05.txt`
(verified, not inferred).

**What it tests:** F3-Net's (ECCV 2020) headline claim that its frequency-aware
decomposition helps most on low-quality/heavily-compressed media. Our previous
evidence was c23-only, so this closes the one axis the whole trade-off framing
rests on.

## Headline result

3-seed mean ROC-AUC:

| config | c40 | c23 | compression cost |
|---|---|---|---|
| baseline_spatial (ResNet-18) | 0.98783 | 0.99487 | −0.00704 |
| xception | 0.99331 | 0.99765 | −0.00434 |
| f3net (Xception + FAD) | 0.99345 | 0.99709 | −0.00364 |

**f3net − xception at c40: +0.00014 (p = 0.2283, not significant).**
Per-seed paired differences: **+0.00024, +0.00019, −0.00002** — straddles zero.

Cohen's d(paired) = +0.99, which looks "large" but is misleading here: the
between-seed variance is itself tiny (sd = 0.00014), so a standardised effect size
inflates a difference of **0.014 AUC points**. Report the absolute magnitude
alongside d, never d alone.

## Reading it

1. **F3-Net's c40 advantage does not replicate under matched training.** The gap is
   +0.014 AUC points and not significant across 3 seeds.
2. **The direction is right, the magnitude is not.** Going c23→c40 the
   f3net−xception gap moves from −0.00056 to +0.00014 (a +0.0007 swing), and F3-Net
   degrades least under compression (−0.00364 vs xception −0.00434). So frequency
   *is* very slightly more compression-robust — consistent with the theory, and
   ~50× too small to justify the architecture.
3. **Backbone matters far more than frequency.** ResNet-18 loses −0.00704 to
   compression; the Xception-family models lose ~−0.004. The choice of backbone
   buys ~5× more compression robustness than adding FAD does.

## ⚠️ Critical caveat — do not write this up as-is

These are **L1 frame-level splits** (crops from the same video in train *and* test —
see `docs/validation-plan.md` C0). Absolute AUCs here (~0.993 at c40) are far above
F3-Net's published c40 figures (~0.90–0.96), which indicates a substantially easier
evaluation, not better modelling.

**This matters most for exactly this result.** At 0.993 AUC there is almost no
headroom, and a null result measured at ceiling is observationally identical to a
real difference being masked by that ceiling. The finding we obtained is the one
that most needs re-verification under C0's video- and identity-level splits before
it can be claimed. Treat this table as the **L1 baseline for the leakage
quantification (C0.3)** and as strong-but-provisional evidence — not as the final
answer.

## Files

| file | contents |
|---|---|
| `summary.csv` | one row per (config, seed) — all 6 metrics |
| `ablation_c40.md` | mean ± std + paired-t p-values vs `xception` |
| `ablation_table.csv` | same, machine-readable |
| `ablation_auc.png` | bar chart with error bars |
| `per_run/*.json` | raw `test_metrics.json` for each of the 9 runs |

Checkpoints (9 × `best.pt`, ~570 MB) are not in git — local backup at
`~/Downloads/c40_results.tar.gz` plus the Kaggle version output.
