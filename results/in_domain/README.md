# c23 in-domain matrix — Phase 2, crop-randomised (L1)

⚠️ **Crop-randomised splits.** These 30 runs used `--group-by None`, so test crops
come from videos that also appear in training. This is the leaky protocol the
paper exists to criticise, and it is why absolute AUCs here reach 0.99+.

**Their role in the current paper is as the L1 arm of contribution 1** — the
protocol gap is the difference between these and `../in_domain_c23_vid/`. Do not
cite the absolute numbers as performance.

⚠️ Crop-level predictions were **not retained** from these runs (they predate
`src/predict.py`), so only summary metrics survive. That is why the protocol gap
can only be reported on the frame-pooled estimand, and why the effect of video
aggregation on its magnitude or direction is untested — see ESSENCE §8a item 5.

10 configurations × 3 seeds. `summary.csv`, `ablation_table.csv/.md`,
`per_run/*.json`, plus `ffppfix_*` — the 2-run weight-decay control that ruled out
the AdamW-on-`alpha_logit` confound (`7bb4e11`).
