# Methods notes — the four questions a marker asks

Answers grounded in what was done, each traceable. Written 2026-09-21/25.

---

## 1 · Efficiency and cost

Measured by `eff_table.py` (fvcore `FlopCountAnalysis`, 50 timed iterations after
5 warmup, CUDA-synchronised). `results/analysis/efficiency/params_flops_latency.csv`.

| | params | GFLOPs | batch-32 latency |
|---|---|---|---|
| Xception | 20.81 M | 1.48 | 40.7 ms |
| Xception + FAD | 20.86 M | 1.52 | 42.0 ms |
| **FAD adds** | **+0.24%** | **+2.70%** | **+3.19%** |

✅ *"FAD costs ≈ +3%"* — true of **GFLOPs and latency**, not parameters (+0.24%).
**Name the quantity when quoting it.**

**Why the cost number matters to the argument.** It makes the finding interesting
rather than deflating: not an expensive component failing to pay off, but a **3%
component whose benefit is smaller than the evaluation can resolve.**

⚠️ The separate-frequency-branch figures (`baseline_spatial` → `full`) are
**disputed**: the README says +44% latency, its own CSV gives **+46.1%**. Do not
quote until reconciled — the CSV is the generated artifact and should win.

---

## 2 · Hyperparameters — what they are

`results/reference/train_args_c40_vid.json`:

15 epochs · batch 64 · **lr 3×10⁻⁴** · weight decay 0.05 · 128 px · AdamW ·
focal loss (γ = 2.0, α = 0.200) · grad clip 1.0 · **fp32, no AMP** ·
ImageNet-pretrained · `--group-by videos-([0-9]+)`

Every training command is **built from that file**; a parameter it contains that a
command fails to pass is a **fatal error**. That guard exists because passing
`--amp` once cost a 155-minute run.

---

## 3 · How they were chosen — and the one departure

**No hyperparameter search was performed.** These are the script defaults.

**Why defensible:** the comparison is **matched, not optimised**. Both arms receive
an identical prespecified recipe, so the difference between them is not confounded
by unequal tuning effort. Tuning one arm alone would build the answer into the
design.

**What it costs:** no model is claimed to be at its best achievable performance.
Absolute AUCs are **not competitive numbers** and are never presented as such.

### ⚠️ Focal loss — where we match the source and where we do not

| | Lin et al. (ICCV 2017) | ours | |
|---|---|---|---|
| formula | FL(p_t) = −α_t(1 − p_t)^γ log(p_t) | identical | ✅ |
| **γ** | *"γ = 2 (our default setting)"* | **2.0** | ✅ **matches** |
| **α** | **0.25** at γ = 2 | **0.200** | ⚠️ **differs** |

Our α = `n_real / (n_real + n_fake)` = 4800/24000, applied to the **positive class
(fake)**, so α_real = 0.80 — giving **equal aggregate class weight**:
19,200 × 0.20 = 4,800 × 0.80 = **3,840**.

❌ Never write *"following Lin et al."* unqualified — true of γ, false of α.

---

## 4 · Validation — how settings were checked

**Checkpoint selection.** A held-out validation split (10%), never trained on. Best
epoch by **validation ROC-AUC**, the **same rule for both arms**. The test split is
scored **once**.

**The sensitivity attempt, and its outcome.** C1 swept lr ∈ {1e-4, 3e-4, 1e-3}
across **both** arms, prespecified before running. It was **uninformative**, and the
replication is what established that:

| lr | sweep 1 | sweep 2 |
|---|---|---|
| 1e-4 | +0.0064 | **−0.0084** |
| 3e-4 | +0.0151 | **−0.0062** |
| 1e-3 | Xception diverged to exactly 0.5000 — **in both sweeps** |

Signs flipped at both usable rates; five of six cells in sweep 2 peaked at the
final epoch. Our own prespecification forbids extending the budget after seeing
results.

❌ **C1 cannot be used to justify 3×10⁻⁴.** It is supplementary, a **failed
sensitivity audit**.
✅ One thing it did establish: **divergence at 1e-3 reproduces exactly**, so that
rate is unusable for this configuration.
✅ **The fairness argument stands without it:** both architectures received the
same prespecified recipe.

---

## 5 · Ablation

**Ours** — the single-component comparison: Xception vs Xception+FAD, matched
backbone and recipe, five training runs on a fixed split.

**Theirs** — F3-Net Fig. 7(a) adds FAD, LFS and MixBlock one at a time:

| row | configuration | AUC | gain |
|---|---|---|---|
| 1 | Xception | 0.893 | — |
| **2** | **+ FAD** | **0.907** | **+0.014** ← our reference |
| 3 | + LFS | 0.920 | +0.027 |
| 4 | + FAD + LFS | 0.928 | +0.035 |
| 5 | full F3-Net | 0.933 | +0.040 |

**We use row 2 − row 1 because we implemented FAD only.** Using +0.040 would have
let us claim **exclusion** — our interval's upper end is +0.0180 — so we chose the
number that **denies us the stronger claim**.

Two observations from their own table, both ours rather than theirs: **LFS
contributes ~1.9× what FAD does**, and the branches are **sub-additive**
(0.014 + 0.027 = 0.041 against a combined +0.035).

---

## 6 · Prespecification

Decision rules committed to version control **before** the analyses they govern —
four times (`crossed`, `target-group`, `c1-lr`, `c2-repeat`).

**They cost us two experiments**, which is the evidence they were real:

- C1's *"a cell peaking at the final epoch is inconclusive"* → five of six cells
- C1's *"do not extend the epoch budget"* → blocked the obvious fix
- C2's *"no causal attribution, under any outcome"* → still binds, though the
  session pattern looks systematic

**Effect thresholds are external**: +0.014 from F3-Net's published ablation, never
selected from our own results.

✅ *"Decision rules were committed before the analyses they governed. They excluded
two experimental cells and prevented a re-run that would otherwise have been
attempted. The principal results were unaffected."*

That last sentence is only credible because of the first two.
