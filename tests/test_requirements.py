"""Every third-party import in src/ must be declared in requirements.txt.

The paper's contribution includes a reproducibility spine, so an environment
built from requirements.txt must actually be able to run the pipeline. It could
not: pillow (every crop is loaded through PIL) and tensorboard (imported at
src/train.py module scope) were both undeclared, so a fresh environment failed
at import time before training began. Found while rehearsing the NPR campaign
locally, not by reading the file.

`torch.utils.tensorboard` is the case a naive scan misses -- its top-level module
is `torch`, which IS declared, while the real requirement is a separate package.
It is checked explicitly below for that reason.
"""
from __future__ import annotations

import ast
import importlib.util
import pathlib
import sys
import sysconfig

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
REQ = pathlib.Path(__file__).resolve().parent.parent / "requirements.txt"

# Import name -> distribution name on PyPI.
DIST = {"sklearn": "scikit-learn", "cv2": "opencv-python", "PIL": "pillow",
        "yaml": "pyyaml", "facenet_pytorch": "facenet-pytorch", "umap": "umap-learn"}


def _requirements() -> str:
    return REQ.read_text().lower()


def _is_stdlib(name: str) -> bool:
    """Resolve by path rather than sys.stdlib_module_names, which needs 3.10+."""
    if name in sys.builtin_module_names:
        return True
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return False
    if spec is None or not spec.origin:
        return False
    if spec.origin in ("built-in", "frozen"):
        return True
    stdlib = sysconfig.get_paths().get("stdlib", "")
    return bool(stdlib) and spec.origin.startswith(stdlib) and "site-packages" not in spec.origin


def _top_level_imports() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    found.setdefault(alias.name.split(".")[0], set()).add(path.name)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.setdefault(node.module.split(".")[0], set()).add(path.name)
    return found


def test_every_third_party_import_is_declared():
    req = _requirements()
    undeclared = {}
    for mod, users in _top_level_imports().items():
        if mod == "src" or _is_stdlib(mod):
            continue
        dist = DIST.get(mod, mod)
        if dist.lower() not in req:
            undeclared[dist] = sorted(users)
    assert not undeclared, f"undeclared in requirements.txt: {undeclared}"


def test_tensorboard_is_declared_even_though_it_is_imported_through_torch():
    """src/train.py:19 does `from torch.utils.tensorboard import SummaryWriter`,
    whose top-level module is torch. Training dies at import without the
    separate tensorboard package, and no scan of top-level names will say so."""
    train = (SRC / "train.py").read_text()
    if "torch.utils.tensorboard" in train:
        assert "tensorboard" in _requirements(), (
            "src/train.py imports torch.utils.tensorboard but requirements.txt "
            "does not declare tensorboard")


def test_pillow_is_declared_because_the_data_loader_needs_it():
    data = (SRC / "data.py").read_text()
    if "PIL" in data:
        assert "pillow" in _requirements()
