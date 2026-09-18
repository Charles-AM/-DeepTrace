# Aggregation space — mean-logit vs mean-probability (post-hoc)

⚠️ **POST-HOC SENSITIVITY ANALYSIS.** The ambiguity was discovered while validating
contribution 2's numbers on 2026-09-17; the analysis was specified after. It is
**not** prespecified and must never be presented as if it were.

⚠️ **The frozen primary result is unchanged.** `results/canonical.json` stands.
Nothing here revises it.

## What this tests

"Video-aggregated AUC" does not fully specify an estimand. Frame scores can be
averaged in **logit space** — what `src/cluster_boot.py` does, and what produced the
canonical result — or in **probability space**. Averaging and the sigmoid do not
commute, so the two can rank videos differently and yield different AUCs. Neither is
incorrect; they encode different aggregation rules.

No paper in `docs/split-policy-audit.md` specifies which it uses.

## Result

V8 fixed-split c40, 5 training runs, 29 components, 150 videos. **Paired**: one set
of 50,000 component × training-run resamples generated once and applied to **both**
arms, so the contrast carries no Monte Carlo noise between them.

| aggregation space | point | 95% CI | half-width | P(> +0.014) | contains 0 | contains +0.014 |
|---|---|---|---|---|---|---|
| **mean-logit** (canonical) | −0.0092 | [−0.0360, +0.0175] | 0.0268 | **0.0405** | yes | yes |
| **mean-probability** | −0.0060 | [−0.0379, +0.0238] | 0.0308 | **0.0825** | yes | yes |

**Paired per-replicate difference** (probability − logit): mean **+0.00286**, 95% of
replicates within [−0.0072, +0.0126]. Not a constant offset — it moves individual
resamples in both directions.

Under the **same** resample, the two rules disagree about whether that replicate
exceeds +0.014 in **4.48%** of cases.

## Reading

**Robust conclusion, aggregation-sensitive quantities.** Both rules give the same
substantive answer: the interval contains zero and +0.014, so the evaluation cannot
distinguish no advantage from a gain of the published magnitude. But the point
estimate moves 0.0032, the interval widens ~15%, and **the exceedance proportion
doubles** — and that proportion is quoted in the manuscript.

⚠️ 4.05% and 8.25% are **proportions of bootstrap replicates exceeding the reference
effect**, not probabilities that FAD works.

## Relation to the frozen result

The mean-logit row agrees with `canonical.json` **within Monte Carlo error**, not
exactly — the frozen run's resampling indices were not retained, so an independent
implementation cannot reproduce it exactly. Exact reproduction requires rerunning
`src/crossed_boot.py`, which was done on 2026-09-17 and matched (ledger §18.1).

## Reproduce

CPU, ~2 minutes, from committed prediction dumps only. See ledger §19 for the
method; `agg_space_c40_v8.csv` holds the figures above.

## Which contribution

**Contribution 1**, the estimand half — it defines the video *score*. Not
contribution 2, which concerns how *uncertainty* is computed. Conflating them would
undo the separation in ledger §10.
