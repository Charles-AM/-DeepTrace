# ⛔ DO NOT CITE — computed on L1 (leaky) checkpoints

**Gap G7.** These results came from the pre-protocol-fix `c40-run` output, i.e.
crop-randomised checkpoints — models evaluated on frames from videos they trained
on, the exact protocol this project exists to criticise.

`src/late_fusion.py` reloads `best.pt`, so it is affected.

**Redoing it requires c23.** It needs `baseline_spatial` + `frequency_only`, and
only c23 has both at video-disjoint (L2); the c40 video-level campaign trained
`baseline_spatial`, `xception` and `f3net` only. ~15 min GPU, 3 seeds.

⚠️ Under review for **removal from the paper entirely** rather than redoing:
it introduces a third architecture pair that is not the case study, and the
complementarity question is not load-bearing for any of the three contributions.
See `../../docs/ESSENCE.md` §12.
