# V6 — threshold sensitivity

Equivalence and exclusion flags at margins 0.005 / 0.010 / 0.014 / 0.020, all
scored off the **same** bootstrap distribution so they are mutually consistent.
8 runs (c40 seeds 0–4, c23 seeds 0–2) × 2 units.

`equivalent_m` (the whole interval inside ±m) and `excludes_m` (upper bound below
+m) are **different claims** and are reported separately. Our c40 component result
[−0.0534, +0.0096] satisfies the second and fails the first; collapsing them is how
"not significant" becomes "no effect".

Interpretation and the summary table: `../resolution/README.md` §5.

```
python -m src.cluster_boot --a <preds_a> --b <preds_b> --margins 0.005 0.010 0.014 0.020 --out-dir results/analysis/thresholds
```
