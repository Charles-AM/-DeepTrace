# Running DeepTrace on Kaggle Notebooks

Local disk is tight, so **all training and dataset work happens on Kaggle** (free
P100 / T4 x2, ~30 GPU-hrs/week). Your Mac is only for editing code and running the
unit tests. This guide gets a notebook running end to end.

---

## 1. One-time account setup

1. Create an account at <https://www.kaggle.com>.
2. **Verify your phone number**: *Settings → Phone Verification*. Required to enable
   GPU **and** internet access in notebooks.
3. (Optional) *Settings → Account → Create New API Token* downloads `kaggle.json`.
   Only needed if you want to push/pull datasets from the CLI. **Never commit it.**

## 2. Create the notebook

1. *Create → New Notebook*.
2. Right sidebar → **Session options**:
   - **Accelerator:** `GPU P100` (or `GPU T4 x2`).
   - **Internet:** `On` (needed to `git clone` and `pip install`).
   - **Persistence:** `Files only` so `/kaggle/working` survives between sessions.

## 3. Pull the code

First cell — this is safe to re-run (clones once, then just pulls). The `./` before
`-DeepTrace` matters: the leading dash in the repo name confuses `%cd` and most
shell commands otherwise.

```python
import os
if not os.path.isdir("-DeepTrace"):
    !git clone https://github.com/Charles-AM/-DeepTrace.git
%cd ./-DeepTrace
!git pull -q
!pip -q install -r requirements.txt   # most deps already present on Kaggle
```

You only ever **pull** on Kaggle. Editing and committing happens on your Mac.

## 4. Attach datasets

Right sidebar → **Add Input** → search and add:

| Need | What to search | Notes |
|------|----------------|-------|
| Early pipeline smoke test | `140k real and fake faces` | ~4 GB, instantly available, permissively hosted. GAN faces, not face-swap — fine for wiring things up. |
| FaceForensics++ | `faceforensics` | Prefer the **official** download once your access request is approved (upload it as a private Kaggle dataset). Community mirrors exist but are EULA-gray. |
| Celeb-DF v2 | `celeb-df` | For the cross-dataset test (Task 7). |

Attached datasets mount read-only under `/kaggle/input/<slug>/`. They do **not**
count against your disk.

## 5. Run a task

```python
!pytest -q tests/test_dct.py            # Task 1 sanity check
# later: !python -m src.train --config configs/full.yaml --data /kaggle/input/...
```

## 6. Keep your outputs

- Checkpoints / CSVs: write them to `/kaggle/working/` (persisted with *Files only*).
- To keep them long-term or move between notebooks: *Save Version* (commits the
  whole notebook + `/kaggle/working`), or *Output → New Dataset*.
- Download a single file from the notebook's **Output** tab.

## 7. Typical GPU budget for the scoped plan

| Job | Approx. GPU time |
|-----|------------------|
| One training run (128px, ~20k imgs, 15 epochs, ResNet-18) | 20–40 min |
| Full ablation matrix (5 configs x 2 seeds) | ~4–6 hrs |
| JPEG robustness sweep (eval only) | ~1 hr |
| Cross-dataset eval | ~15 min |

Comfortably inside the weekly free quota if spread across a few sessions.
