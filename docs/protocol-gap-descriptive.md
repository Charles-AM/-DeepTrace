# Protocol gap, per architecture and seed — descriptive

⚠️ **DESCRIPTIVE ONLY. NOT PRESPECIFIED. NOT AN UNCERTAINTY ESTIMATE.**
Reproduce with `python -m src.protocol_gap` (CPU, instant, committed CSVs only).

Strengthens research question 1 without new compute. The L1 campaign predates the
artifact-retention policy, so no prediction dumps exist and no cluster-aware
interval is possible (ledger §25). What the committed per-run summaries *do* allow
is reporting the gap across every architecture and seed rather than as one number.

## 1. Protocol gap — L1 crop-randomised minus L2 video-disjoint

Frame-pooled ROC-AUC, c40, corresponding seeds 0, 1, 2.

| architecture | per-seed gap | mean | run-to-run sd |
|---|---|---|---|
| baseline_spatial | 0.1928, 0.1866, 0.1968 | **+0.1921** | 0.0051 |
| xception | 0.1811, 0.1405, 0.2077 | **+0.1764** | 0.0338 |
| xception_fad | 0.2009, 0.1509, 0.1934 | **+0.1817** | 0.0270 |

**Defensible sentence:**

> Across three seeds and three architectures, crop-randomised evaluation reported
> frame-pooled AUC values 0.141 to 0.208 higher than video-disjoint evaluation. The
> mean protocol gaps ranged from 0.176 to 0.192 AUC, showing that the large
> difference was not driven by one architecture or one seed.

The smallest single observation, 0.1405, is still **10× the +0.014 reference**.

## 2. FAD minus Xception, within each split

| split | n | per-seed difference | mean | run-to-run sd |
|---|---|---|---|---|
| L1 crop-randomised | 3 | +0.0002, +0.0002, −0.0000 | **+0.0001** | 0.0001 |
| L2 video-disjoint | 5 | −0.0196, −0.0101, +0.0142, +0.0253, +0.0145 | **+0.0049** | 0.0188 |

**Defensible sentence:**

> The mean FAD minus Xception difference remained small under both protocols,
> +0.0001 for crop-randomised evaluation and +0.0049 for video-disjoint evaluation.
> Run-to-run spread was much smaller under crop randomisation, but this comparison
> is descriptive because L1 used only three runs, operated near the AUC ceiling, and
> lacks prediction-level outputs for cluster-aware uncertainty.

This is *the conclusion was stable; its warrant was not*, with numbers under it.

---

## What the sd is, and is not

| | |
|---|---|
| **is** | run-to-run spread across training seeds |
| **is not** | test-content uncertainty — that needs the prediction dumps L1 lacks |
| **is not** | comparable between campaigns: 3 seeds against 5 |

**"Corresponding seeds", never "paired."** A shared seed fixes initialisation and
data order. It does **not** make the two test sets the same — L1 and L2 partition
the data differently, which is the entire point of the comparison. Calling it paired
implies a statistical structure that is not present.

## What must not be concluded

❌ **"L1 was more precise."** L1 sits near the ceiling (~0.993–0.995), which
compresses differences mechanically, and rests on 3 seeds against L2's 5. ESSENCE
forbids claiming L1 could not have detected a difference — near-ceiling AUC does not
prove low power. The tight spread is *consistent with* ceiling compression and does
not demonstrate false precision.

❌ **"The gap measures leakage."** L1 against L2 changes prior exposure **and** the
test partition together. **V2 is the experiment that separates them and keeps the
headline for that question.**

✅ The claim this does support, which generalises past L1:

> **Agreement across training seeds can create an appearance of stability while
> leaving test-content uncertainty unmeasured.**

That applies to any paper reporting mean ± seed standard deviation and stopping
there.

## Naming

The FAD arm is `xception_fad`. The `f3net` string in `results/in_domain_c40*/summary.csv`
is an alias retained only because it is embedded in artifact filenames produced
before 2026-09-06 (`src/config.py`). `src/protocol_gap.py` maps it on read.
**Never print either as "F3-Net"** — this is the FAD branch alone, no LFS, no
MixBlock, and the full system's figure is 0.040 rather than 0.014.
