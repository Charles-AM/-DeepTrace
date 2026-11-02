# Two executions of the NPR campaign — observations

**Recorded 2026-11-01.** Campaign 1 (`preds_npr`) and campaign 2 (`preds_npr2`) ran
the same frozen recipe, the same committed order table and the **same runner file**
on separate Kaggle instances. Campaign 2's clone was asserted at `4339b87`.

🚨 **These are frame-pooled `test_auc` values printed by the training loop, not
estimates.** No bootstrap has been run. The video-aggregated, component-aware,
conditional and crossed analyses are still pending and will not reproduce these
numbers. Nothing here may be reported as an effect estimate.

Both campaigns passed the pre-analysis gate independently. Campaign 2's gate
confirmed containment in `preds_npr2`, one `split_seed`, one 3000-item test set and
one 29-component partition.

## Per-run results

| seed | Xception c1 | Xception c2 | NPR c1 | NPR c2 |
|---|---|---|---|---|
| 0 | 0.8148 | 0.7934 | **0.5000** | **0.5000** |
| 1 | 0.7956 | 0.7817 | 0.7978 | 0.7949 |
| 2 | 0.8017 | 0.7966 | 0.7965 | 0.7783 |
| 3 | 0.7690 | 0.7983 | 0.7780 | 0.7728 |
| 4 | 0.7988 | 0.8018 | 0.7894 | 0.7650 |

## 1. The seed-0 collapse is reproducible across instances

`xception_npr` seed 0 produced `train_loss=0.0555` unchanged for all 15 epochs and
`test_auc=0.5000 test_acc=0.8000 test_f1=0.8889 test_eer=0.5000` — identical to
campaign 1, and identical to what a constant predictor returns through
`src/metrics.py`. Counting the two timing runs, campaign 1, the diagnostic control
and campaign 2, that is **5/5 across at least three physical instances**.

## 2. Basin entry is partly, not wholly, seed-determined

`xception_seed3` entered the same region in **both** campaigns and escaped both
times, at the same point:

| | epoch 0 | epoch 1 | epoch 2 |
|---|---|---|---|
| campaign 1 | 0.0555 / auc 0.5170 | 0.0554 / 0.5866 | escaped |
| campaign 2 | 0.0555 / auc 0.5083 | 0.0554 / 0.5565 | escaped |

But `xception_npr_seed1` began campaign 2 inside the region
(`train_loss=0.0556, val_auc=0.5187`) and began campaign 1 outside it
(`0.0549, val_auc=0.6683`), escaping either way. So entry replicates at some seeds
and not others, and **"deterministic" applies to the seed-0 NPR outcome, not to
basin entry in general.**

## 3. The paired architectural difference is not stable across executions

This is the measurement campaign 2 was run for. Seed 0 is excluded because NPR is
pinned at exactly 0.5 in both executions, so its movement is an artifact of the
Xception arm alone.

| seed | NPR − Xception, c1 | NPR − Xception, c2 | moved | sign flip |
|---|---|---|---|---|
| 1 | +0.0022 | +0.0132 | +0.0110 | |
| 2 | −0.0052 | −0.0183 | −0.0131 | |
| 3 | +0.0090 | −0.0255 | **−0.0345** | **yes** |
| 4 | −0.0094 | −0.0368 | −0.0274 | |

- Largest single-seed movement of the paired difference: **0.0345**, which is
  **2.5× the 0.014 published reference effect**.
- One of four seeds **flipped sign**.
- Four-seed mean moved from **−0.00085 to −0.01685** — a shift of 0.016, itself
  larger than the effect size this literature reports.
- Within-campaign paired sd nearly tripled: 0.0081 → 0.0214.

This is a paired, same-seed, same-recipe measurement at n=4 (n=5 for the single-arm
movements). It supersedes the n=1 difference-level movement of 0.0336 recorded in
`results/in_domain_c40_fixedsplit/README.md` as the programme's evidence for
execution-level instability — not by contradicting it, but by measuring it properly.

❌ **Still not a between-execution variance.** Two executions give a difference, not
a variance. ❌ Still not a collapse rate.

## 4. The arms are at parity on cost, in two independent campaigns

Campaign 2: every run 33.8–33.9 min for **both** arms (the 49.4 min seed-0 Xception
run is container warmup — it was first). Whole campaign 353.6 min = **5.89 h**,
against campaign 1's 5.94 h. Two independent campaigns show no NPR wall-clock
penalty, which closes the question the non-replicating +15.7% probe left open.

## Why campaign 1 remains load-bearing

The result in §3 **is** the two-campaign comparison. It cannot be computed from
campaign 2 alone. Campaign 1 is not a spoiled run that campaign 2 supersedes; it is
half of the programme's strongest reproducibility measurement.
