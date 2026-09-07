# V4 / C0b — the aggregation effect

Frame-pooled vs video-level AUC on the **same** held-out videos, c40 seeds 0–4.
Same model, same test set, two aggregations.

Video-level aggregation moved the estimate in all five seeds while preserving
sign (shifts −0.0107 to +0.0061). Aggregation therefore defines the **estimand**,
not merely the precision — it is not one of the uncertainty sources in
`../../docs/ESSENCE.md` §0b.

Table: `../resolution/README.md` §5.

```
python -m src.cluster_boot --a <preds_a> --b <preds_b> --margins 0.014 --out-dir results/analysis/aggregation
python -m src.cluster_boot --a <preds_a> --b <preds_b> --frame-level --margins 0.014 --out-dir results/analysis/aggregation
```
