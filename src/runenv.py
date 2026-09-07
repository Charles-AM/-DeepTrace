"""Capture everything needed to tell a repeatability finding from a config slip.

V8 showed a nominally identical seed-0 run moving the FAD-Xception difference by
0.034. That is only interesting if the two runs really were identical. The first
V8 attempt was *not* -- it passed `--amp` -- and that cost a 155-minute run before
anyone noticed. Nothing in the artifacts would have revealed it; the difference
lived in a script.

So every run that participates in a repeatability claim records what it actually
ran under. A later diff of two of these files answers "genuine nondeterminism or
an unnoticed difference?" without re-deriving anything.

Captured: git commit and cleanliness, GPU model, torch/CUDA/cuDNN versions,
determinism and TF32 settings, the environment variables that change numerics, the
manifest hash, and an initialisation probe.

The probe reproduces train.py's init exactly -- `seed_everything(seed)` then
`build_model(...)`, which is the first torch RNG consumer in that script -- and
hashes the resulting parameters. Identical hashes prove weight initialisation
matched; differing hashes localise the problem to setup rather than training.

    python -m src.runenv --tag repeat_rep1 --manifest results/manifests/x.csv \\
        --probe-config xception --out-dir results/reference
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path


def hash_file(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while (b := fh.read(chunk)):
            h.update(b)
    return h.hexdigest()


def git_state(repo=".") -> dict:
    def sh(*a):
        try:
            out = subprocess.run(["git", "-C", str(repo), *a], capture_output=True,
                                 text=True, check=True).stdout
            # coerce: a caller that patched subprocess (or a text=False build) can
            # hand back bytes, and this record must stay JSON-serialisable
            if isinstance(out, bytes):
                out = out.decode("utf-8", "replace")
            return str(out).strip()
        except Exception:
            return "unknown"
    return {"commit": sh("rev-parse", "HEAD"),
            "dirty": bool(sh("status", "--porcelain")),
            "branch": sh("rev-parse", "--abbrev-ref", "HEAD")}


def torch_state() -> dict:
    """Numerics-relevant runtime settings. TF32 in particular silently changes
    matmul precision on Ampere and later, and is on by default in some builds."""
    try:
        import torch
    except ImportError:
        return {"torch": "not installed"}
    out = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda": getattr(torch.version, "cuda", None),
        "cudnn": (torch.backends.cudnn.version()
                  if torch.backends.cudnn.is_available() else None),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    if torch.cuda.is_available():
        out["gpus"] = [torch.cuda.get_device_name(i)
                       for i in range(torch.cuda.device_count())]
        out["gpu_count"] = torch.cuda.device_count()
    return out


def probe_init_hash(config: str, image_size: int, seed: int) -> str | None:
    """Hash the model parameters immediately after initialisation.

    Mirrors train.py's order: seed_everything(seed) then build_model. Nothing
    between them consumes torch RNG there, so this reproduces the exact initial
    weights that run started from.
    """
    try:
        import torch
        from .config import build_model
        from .utils import seed_everything
    except Exception:
        return None
    seed_everything(seed)
    model = build_model(config, image_size=image_size, pretrained=True)
    h = hashlib.sha256()
    for name, p in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


ENV_KEYS = ("CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED", "CUDA_VISIBLE_DEVICES",
            "OMP_NUM_THREADS", "TORCH_CUDNN_V8_API_ENABLED")


def capture(tag: str, manifest=None, probe_config=None, image_size=128, seed=0,
            extra: dict | None = None, repo=".") -> dict:
    rec = {"tag": tag, "platform": platform.platform(),
           "python": platform.python_version(),
           "git": git_state(repo), "torch": torch_state(),
           "env": {k: os.environ.get(k) for k in ENV_KEYS}}
    if manifest and Path(manifest).exists():
        rec["manifest"] = {"path": str(manifest), "sha256": hash_file(manifest)}
    if probe_config:
        rec["init_probe"] = {"config": probe_config, "seed": seed,
                             "image_size": image_size,
                             "param_sha256": probe_init_hash(probe_config, image_size, seed)}
    if extra:
        rec["extra"] = extra
    return rec


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", required=True)
    p.add_argument("--manifest", default=None)
    p.add_argument("--probe-config", default=None)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repo", default=".")
    p.add_argument("--out-dir", default="results/reference")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    rec = capture(a.tag, manifest=a.manifest, probe_config=a.probe_config,
                  image_size=a.image_size, seed=a.seed, repo=a.repo)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / f"runenv_{a.tag}.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))
    print(f"\nwrote {out}/runenv_{a.tag}.json")


if __name__ == "__main__":
    main()
