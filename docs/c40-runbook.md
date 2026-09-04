# c40 runbook — first session after Kaggle quota resets

**Goal (C7):** reproduce F3-Net's headline claim — that its frequency features help
most on heavily-compressed (c40) video — under matched training. If `f3net` does not
significantly beat `xception` at c40, the paper's spine holds.

Run it as **one Save Version → "Save & Run All (Commit)"** notebook so it survives
the browser tab closing. ~5 h. Accelerator: **GPU T4 x2** (never P100).

## Notebook cells

**Cell 1 — clone**
```bash
!cd /kaggle/working && rm -rf ./-DeepTrace && git clone -q https://github.com/Charles-AM/-DeepTrace.git ./-DeepTrace && cd ./-DeepTrace && git log --oneline -1
```

**Cell 2 — deps** (do NOT `pip install -r requirements.txt` — it breaks Kaggle's torch)
```bash
!pip -q install --no-deps facenet-pytorch
```

**Cell 3 — download FF++ at c40** (same 150 pairs as c23 → same videos → same split)
```bash
!cd /kaggle/working/-DeepTrace && python -m src.ffpp_fastdl /kaggle/working/ffpp_c40 --originals --num-pairs 150 --compression c40 --workers 16
```
c40 files are ~3–5× smaller than c23 → ~15–25 min.

**Cell 4 — extract faces** (same params as the c23 set: 160 px, every 12th frame, ≤20/video)
```bash
!cd /kaggle/working/-DeepTrace && python -m src.extract_faces_v2 --videos /kaggle/working/ffpp_c40 --out /kaggle/working/ffpp_c40_crops --size 160 --every 12 --max-per-video 20 --device cuda
```
~40–60 min. Expect ~30k crops (6k real / 24k fake). Sanity-check the printed
real/fake counts before training.

**Cell 5 — train the matched comparison** (3 configs × 3 seeds, 15 epochs)
```bash
!cd /kaggle/working/-DeepTrace && python -m src.run_ablation --data-root /kaggle/working/ffpp_c40_crops --dataset-name ffpp_c40 --configs baseline_spatial xception f3net --seeds 0 1 2 --epochs 15 --image-size 128 --batch-size 64 --reference xception
```
~3.5–4 h (baseline_spatial ~12 min, xception ~30 min, f3net ~35 min, ×3 seeds).
Output: `results/ablation_table.md` (rename to `ablation_c40.md` after), `summary.csv`,
30 `results/ffpp_c40_*/` run dirs with `best.pt` + `test_metrics.json`.

**Cell 6 — bundle + link**
```bash
!cd /kaggle/working && cp -r -DeepTrace/results ./c40_results && mv c40_results/ablation_table.md c40_results/ablation_c40.md && tar czf c40_results.tar.gz c40_results && du -h c40_results.tar.gz
```
```python
from IPython.display import FileLink
FileLink('c40_results.tar.gz')
```

## After the run

1. **Download `c40_results.tar.gz`** → send to Claude → integrate into `results/` +
   commit (`results/in_domain_c40/`).
2. **Save the c40 checkpoints as a private Kaggle dataset** `deeptrace-c40-ckpts`
   (New Dataset → from the notebook output, or download the 30 `best.pt` and
   re-upload) — needed later for c40 robustness + c40 spectra. EULA: **private**.
3. Read the result: `f3net` − `xception` AUC at c40, per seed, with the paired-t
   p-value from `ablation_c40.md`.
   - CI includes 0 / not significant → **paper spine confirmed** ("does not
     replicate").
   - `f3net` significantly ahead at c40 → reframe to "we reproduce the c40 gain,
     show it is the only regime it exists and it does not transfer" (still a paper).

## Optional same-session add-on (if quota allows, ~25 min)

**Efficiency table (C8)** — no training:
```bash
!cd /kaggle/working/-DeepTrace && python - <<'PY'
import torch, time
from src.config import build_model
try:
    from fvcore.nn import FlopCountAnalysis
except Exception:
    import subprocess, sys; subprocess.run([sys.executable,"-m","pip","-q","install","--no-deps","fvcore"]); from fvcore.nn import FlopCountAnalysis
dev="cuda"; x=torch.randn(1,3,128,128,device=dev)
for name in ["baseline_spatial","xception","f3net","full"]:
    m=build_model(name,image_size=128,pretrained=False).to(dev).eval()
    p=sum(t.numel() for t in m.parameters())/1e6
    with torch.no_grad(): f=FlopCountAnalysis(m,x).total()/1e9
    xb=torch.randn(32,3,128,128,device=dev)
    with torch.no_grad():
        for _ in range(5): m(xb)
        torch.cuda.synchronize(); t0=time.time()
        for _ in range(50): m(xb)
        torch.cuda.synchronize(); lat=(time.time()-t0)/50*1000
    print(f"{name:18s} params={p:6.2f}M  GFLOPs={f:6.2f}  batch32_lat={lat:6.1f}ms")
PY
```

## Sessions B and C (later weeks)

- **B:** re-extract Celeb-DF v2 (official test list this time) + DFDC → save both as
  **private** Kaggle datasets. Mostly CPU.
- **C:** cross-dataset eval (`src/cross_dataset.py`) of the c23 AND c40 checkpoints
  on Celeb-DF / DFDC / (DF40) × 3 seeds; c40 robustness sweep; c40 spectra.
