# Loss-geometry diagnostic — prespecification

**Status: post hoc follow-up.** This was not planned before the NPR campaign. It
exists because one campaign run (`ffpp_c40_vid_xception_npr_seed0`) converged to a
constant prediction. It is a **diagnostic**, not an experiment about NPR, and it
supports Contribution 3. It does **not** replace or amend the FAD campaign, which
remains the primary architectural case study.

🚨 **These three runs may never enter any arm comparison or any NPR effect
estimate.** They train the NPR arm only, at one seed, under two deliberately
altered losses. They are not campaign runs and they are not replacements for
campaign runs. They write to `results_diag/` and `preds_diag/`, which no analysis
glob touches.

## What was observed

`xception_npr` seed 0 produced `train_loss=0.0555` unchanged for all 15 epochs,
`test_auc=0.5000`, `test_acc=0.8000`, `test_f1=0.8889`, `test_eer=0.5000`. Running
a constant predictor through `src/metrics.py` returns `roc_auc 0.5, accuracy 0.8,
f1 0.88889, eer 0.5` — an exact match on all four. The run converged to a constant
prediction.

The same configuration collapsed 3/3 times at seed 0 (two timing runs and the
campaign run). Seeds 1–4 trained normally. `xception_seed3` entered the same region
(`train_loss=0.0555`, `val_auc=0.5170`) for two epochs and escaped at epoch 2, so
**the region is reached by both arms** and the collapse is not NPR-specific.

## The arithmetic this diagnostic tests

Computed with the repository's own `src/losses.FocalLoss` over the campaign's class
counts (4800 real, 19200 fake):

| loss | best constant prediction | loss there |
|---|---|---|
| focal γ=2 α=0.2 — **the recipe** | p(fake)=**0.500** (on the decision threshold) | 0.0555 |
| weighted CE γ=0 α=0.2 | p(fake)=**0.500** | 0.2218 |
| focal γ=2 α=0.5 | p(fake)=**0.627** | 0.0647 |
| plain unweighted CE | p(fake)=0.801 | 0.5004 |

α=0.2 exactly balances the 1:4 ratio: real 4800×0.8 = 3840, fake 19200×0.2 = 3840.

Two supported statements, and no more:

- For **weighted cross-entropy** the best constant prediction is the weighted class
  prior (verified at α = 0.2 / 0.5 / 0.8: 0.500/0.500, 0.801/0.800, 0.941/0.941), so
  balancing the aggregate weights puts it at 0.5 for any imbalance ratio.
- **Focal loss with γ>0 is not a proper loss** and its optimum is *not* the weighted
  prior (0.627 vs 0.800, 0.737 vs 0.941); γ shrinks it toward 0.5. It coincides with
  the weighted prior only at balance, by symmetry.

❌ **Not claimed:** that α *causes* the trapping. The arithmetic locates the best
constant prediction. It does not explain why the network enters the constant-
prediction manifold, and it does not establish a strict local minimum in weight
space. `xception_seed3` escaping proves descent directions out of the region exist.

❌ **Not claimed:** anything about losses outside the two computed here.

## The three runs — one session, seed 0, `xception_npr`

| run | α | γ | best constant prediction | purpose |
|---|---|---|---|---|
| `ctrl` | 0.2 (derived) | 2.0 | 0.500 | does seed 0 still collapse in a *new* session? |
| `alpha` | **0.5** | 2.0 | 0.627 | move the optimum off the decision threshold |
| `gamma` | 0.2 (derived) | **0.0** | 0.500 | remove focusing, keep the optimum at 0.5 |

Everything else comes from `results/reference/train_args_c40_vid.json` unchanged.
The control's command is identical to the campaign's NPR runs apart from
`--out-root` and `--dataset-name`.

🚨 **The control is mandatory.** All three observed collapses carried different
session ids (`S-7c9038c8`, `S-d428bbd4`, `S-29005a04`) but ran inside **one Kaggle
batch execution on one physical instance**. Determinism across instances is
untested, and this repository has already documented a nominally identical seed-0
configuration moving 0.0336 and flipping sign across sessions. Without the control,
an escaping intervention could be the new instance rather than the changed loss.

Order: `ctrl`, `alpha`, `gamma`. The control runs first so a failed premise is known
before any intervention is interpreted. Cost ≈ 3 × 36 min ≈ **1.8 h**.

## Collapse rule — committed before the runs

A run is **collapsed** iff **both** hold:

1. `test AUC ≤ 0.55`
2. `sd(prob_fake) < 0.01` across the 3000 test crops

Accuracy is deliberately excluded. The 0.800/0.200 flipping is an artifact of the
optimum sitting *exactly* on the decision threshold; under any condition whose
optimum is off the threshold a collapsed run shows a stable accuracy and no flip, so
an accuracy-based rule would miss it.

The 0.01 threshold is set from the physics, not from the data: a constant predictor
has `sd = 0` up to float noise, while a discriminating model spreads scores across
much of [0,1]. The runner prints `sd(prob_fake)` for every available dump —
including the ten campaign dumps when they are attached — so the margin is auditable.

**Reported but NOT part of the collapse call:** `mean(prob_fake)` against that
condition's computed optimum (0.500 / 0.627 / 0.500). The location is what the α
intervention manipulates, so it must not gate the collapse decision. It is the
corroborating check on the arithmetic.

## Outcomes, enumerated before the runs

| ctrl | alpha | gamma | reading |
|---|---|---|---|
| collapse | **escape** | collapse | α associated with escape; removing focusing is not |
| collapse | collapse | collapse | α relocates the constant solution (check `mean(prob_fake)` ≈ 0.627) but does not prevent collapse |
| collapse | collapse | **escape** | γ affects optimisation although the optimum stays at 0.5; the α-only account is incomplete |
| collapse | **escape** | **escape** | neither factor is isolated — the collapse is fragile to any loss perturbation, closer to a knife-edge initialisation effect than loss geometry |
| **escape** | any | any | the premise fails: seed 0 is not deterministic across instances. No causal reading of either intervention. Report the non-replication. |

✅ Every outcome is reportable. ❌ No outcome licenses promoting this to a main
contribution unless it is clear **and** independently repeated.

## Isolation

New directories (`results_diag/`, `preds_diag/`). No change to any protected file —
`--focal-alpha` and `--focal-gamma` are existing `src/train.py` flags passed by the
existing `src/reference_cmd.build_train_cmd`.

```bash
git diff main -- src/train.py src/crossed_boot.py src/agg_space.py \
                 src/cluster_boot.py src/clusters.py     # must print nothing
python -m src.reproduce                                   # must return 16/16
```
