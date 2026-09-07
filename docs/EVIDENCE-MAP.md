# Evidence map — every claim traced to code, data, and commit

If asked *"prove it"*, this is the document to hand over. Each row links a claim to
the script that produced it, the committed file holding the numbers, and the commit
that introduced them. Nothing in the paper should make a claim absent from this
table.

Repository: `github.com/Charles-AM/-DeepTrace`. All paths are repo-relative.
Last updated 2026-09-06.

---

## Primary findings

| # | claim | produced by | committed evidence | commit |
|---|---|---|---|---|
| E1 | Crop-randomised splits inflate FF++ AUC by 17.6–19.2 points across three architectures (c40) | `src/run_ablation.py` (L1 vs L2 runs) | `results/in_domain_c40/summary.csv`, `results/in_domain_c40_vid/summary.csv` | `4075973`, `e68b559` |
| E2 | At n=5, FAD − Xception = +0.0049, 95% CI [−0.0185, +0.0283]; excludes neither zero, +0.010, nor +0.014 | `src/paired_summary.py` | `results/in_domain_c40_vid/paired_summary.csv` | `f46d104` |
| E3 | Point estimate sign flips between seed subsets (sensitivity, **not** a frequency) | `src/paired_summary.py` | same | `9e53b05` |
| E4 | Simulated independent 3-seed studies: ~32% report opposite sign, ~20% exceed +0.014 | `src/paired_summary.py` (parametric simulation) | same | `9e53b05` |
| E5 | Backbone comparison is equally unstable (`baseline_spatial − xception`, 9/10 subsets negative) | `src/paired_summary.py` | same | `9e53b05` |
| E6 | FF++ pair graph: 300 sequences → 150 identity components, all size 2 (histogram `2:150`, no singletons, not degenerate). Identity-level bootstrapping has **50% of the units** video-level grouping does | `src/cluster_stats_report.py` | `results/analysis/clusters/component_stats.csv` | `2234a17` |

| E16 | F3-Net's published FAD gain is **+0.014 AUC** (Xception 0.893 → Xception+FAD 0.907) on FF++ LQ/c40; full F3-Net is +0.040. Their FAD is the *learnable* variant (`f_base + f_w`, 3 bands) — fixed filters give only 0.901 — matching our `1[r∈B]+tanh(w)` 3-band implementation | manual verification of the paper | `docs/f3net-ablation-verified.md` | this commit |

| E17 | **Naive frame resampling yields a significant result that correct clustering dissolves.** Same point estimate (−0.0196); frame CI [−0.0325, −0.0079] excludes zero, video CI [−0.0606, +0.0167] and identity CI [−0.0534, +0.0096] do not. SE inflates 3.16× | `src/predict.py` + `src/cluster_boot.py` | `results/analysis/cluster_boot/ALL.csv` | this commit |
| E18 | No cluster-aware interval at any seed or compression excludes zero; 2 of 8 exclude +0.014 | same | same | this commit |
| E19 | Video-level aggregation changes the estimate, not just the interval (−0.0196 frame-pooled vs −0.0303 video-level, same model/test set) | same | same | this commit |

## Supporting analyses

| # | claim | produced by | committed evidence | commit |
|---|---|---|---|---|
| E7 | Late fusion of spatial + frequency-only scores equals spatial alone to 4 dp, 3 seeds, regularised **and** unregularised | `src/late_fusion.py` | `results/analysis/late_fusion/seed_results.csv` | `d5446eb` |
| E8 | CKA(spatial, frequency) ≈ CKA(spatial, random) across 3 seeds | `src/cka.py` | `results/analysis/cka/seed_results.csv` | `d5446eb` |
| E9 | Gated fusion α = 0.496 ± 0.001, never leaves initialisation | `src/gate_readout.py` | `results/analysis/fusion_alpha.csv`, `fusion_alpha_raw.csv` | `c9aefb6` |
| E10 | The α result is not a weight-decay artifact (retrained with the scalar excluded from decay; α unchanged) | `src/utils.py::no_decay_param_groups` + retrain | `results/in_domain/per_run/ffppfix_*.json` | `9deb97f`, `c9aefb6` |
| E11 | Separate frequency branch costs +31% FLOPs / +44% latency; FAD ≈ +3% | `eff_table.py` (listed in `docs/c40-runbook.md`) | `results/analysis/efficiency/params_flops_latency.csv` | `18279e9` |
| E12 | Real-vs-fake DCT gap small (d≈0.15), largely survives JPEG-q30 | `src/spectra.py` | `results/analysis/spectra/summary.csv`, `radial.csv` | `a5bd9fa` |
| E13 | Frequency-only ranks best on the crudest manipulation, worst on the subtlest | `src/permanip.py` | `results/analysis/permanip/` | `a5bd9fa` |
| E14 | Robustness sweep (JPEG/blur/noise/resize/contrast) | `src/robustness.py` | `results/robustness/` | `ca268bc` |
| E15 | In-domain c23 matrix, 10 configs × 3 seeds | `src/run_ablation.py` | `results/in_domain/summary.csv` + 30 `per_run/*.json` | `ca268bc` |

## Methods and environment

| item | location | commit |
|---|---|---|
| Full method, hyperparameters, dataset acquisition, experiment registry | `docs/REPRODUCIBILITY.md` | `4be5338` |
| Exact package versions (torch 2.10.0+cu128 etc.), captured on Kaggle | `docs/pip_freeze_2026-09-05.txt` | `4075973` |
| Experiment counts and compute (~50 runs, ~32 GPU-h) | `docs/EXPERIMENTS.md` | `de83432` |
| Chronological record incl. corrections | `docs/progress-log.md` | multiple |
| Frozen framing, claim-strength discipline | `docs/ESSENCE.md` | `2e92091` |

## Tests

| module | test | verified |
|---|---|---|
| `src/clusters.py` | `tests/test_clusters.py` — reverse-pair merge, transitivity, disjointness, component-id stability, disjoint-vs-cyclic structure | ✅ run locally |
| `src/predict.py` (name parsing) | `tests/test_clusters.py` — `video_id` agrees with the `--group-by` regex capture | ⚠️ needs torch, unrun |
| `src/utils.py` | `tests/test_utils.py` — scalar gates excluded from weight decay; optimiser step leaves `alpha_logit` untouched | ⚠️ needs torch, unrun |
| `src/cluster_boot.py` | exercised end-to-end on synthetic data with sklearn stubbed; **frame-unit CI 2.3× narrower than video-unit** (0.0201 vs 0.0464), degenerate-bootstrap guard raises | ✅ run locally |
| `src/paired_summary.py` | run on real committed data | ✅ |

---

## ⚠️ Gaps — assertions not yet backed by a committed artifact

Every item below is disclosed rather than buried. **None may appear in the paper
without first being closed or explicitly caveated.**

| gap | why it matters | to close |
|---|---|---|
| ~~**G1**~~ — **CLOSED 2026-09-06** | — | Regenerated by `eff_table.py` to file; `frequency_only` row added (0.18M params, 0.18 GFLOPs, 5.2 ms) |
| ~~**G2**~~ — **CLOSED 2026-09-06** | — | Verified against arXiv 2007.09355v2: Fig. 7(a) p.12 and Table 3 p.14. Both Acc and AUC reported; +0.014 is **AUC**. See E16. |
| 🟡 **G3** — `src/predict.py` **now executed and validated** (22 dumps, gates passed). `src/band_ablation.py` still unrun | — | Remaining: smoke-test `band_ablation.py` |
| ~~**G4**~~ — **CLOSED 2026-09-06** | — | `src/cluster_stats_report.py` now generates `results/analysis/clusters/component_stats.csv` from a crop listing |
| ~~**G7**~~ — **RESOLVED 2026-09-06, by removal and replacement** | — | **permanip: replaced** — recomputed from L2 dumps (`results/analysis/permanip_l2/`), no GPU. **late_fusion and cka: REMOVED from the paper** rather than repaired (ESSENCE §12) — they belong to the older "why frequency features fail" framing and would introduce disconnected hypotheses; their directories stay marked DO-NOT-CITE. **spectra: never affected** — it loads no checkpoint. |
| ~~**G5**~~ — **CLOSED 2026-09-06** (as far as the data allow) | — | V8 run: fixed-split training-run sd **0.0145** vs complete-pipeline **0.0188**, implying a split-composition sd of 0.0121 and an illustrative 59/41 allocation. ⚠️ The two variances are **not statistically distinguishable** (F=1.70, df 4,4) and the implied component **changes sign** between procedural attempts, so this is a point estimate, not an established partition. The promised clean separation of "optimisation variability" is withdrawn; see ESSENCE §0b for the vocabulary that replaced it. `results/in_domain_c40_fixedsplit/` |
| ~~**G6**~~ — **CLOSED 2026-09-06** | — | V1 run; see `results/analysis/cluster_boot/`. Frame-unit SE 0.0063 vs video-unit 0.0199 (3.16×) |

## Artifacts held outside git

Checkpoints are excluded by `.gitignore` (size). Locations of record:

| artifact | location |
|---|---|
| c23 30-run checkpoints | Kaggle notebook output `deeptrace3` — **only copy, no local backup** |
| c40 checkpoints + metrics | `~/Downloads/c40_results.tar.gz` (595 MB) + Kaggle version output |
| c40 video-level checkpoints | Kaggle version output |
| Gate-fix checkpoints | `~/Downloads/gatefix_artifacts.tar.gz` (85 MB) |
| FF++ c23 crops | `~/Downloads/ffpp_crops_160.tar.gz` (239 MB) + private Kaggle dataset |
| FF++ c40 crops | Kaggle notebook output `c40-run` |
