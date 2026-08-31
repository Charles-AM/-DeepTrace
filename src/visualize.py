"""Task 8 — explainability & publication figures.

  * learned frequency-mask heatmap
  * radial frequency-importance profile (mask weight vs distance from DC)
  * Grad-CAM overlays on the spatial branch
  * t-SNE (or UMAP) of the fused features, coloured real vs fake

Each function saves a PNG and returns the data it plotted.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .config import build_model
from .data import FaceCropDataset, IMAGENET_MEAN, IMAGENET_STD, _transforms, read_manifest
from .utils import get_device


def _agg_backends():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


# --------------------------------------------------------------------------- --
# 1. learned mask heatmap
# --------------------------------------------------------------------------- --
def mask_heatmap(mask: torch.Tensor | np.ndarray, path: Path) -> np.ndarray:
    """`mask` is (C, H, W) or (H, W); channels are averaged for display."""
    plt = _agg_backends()
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m.mean(axis=0)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(m, cmap="viridis", origin="upper")
    ax.set_title("Learned frequency mask  (DC at top-left)")
    ax.set_xlabel("horizontal frequency →")
    ax.set_ylabel("vertical frequency →")
    fig.colorbar(im, ax=ax, label="keep weight (0–1)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return m


# --------------------------------------------------------------------------- --
# 2. radial frequency importance
# --------------------------------------------------------------------------- --
def radial_profile(mask: torch.Tensor | np.ndarray, path: Path, n_bins: int = 32) -> dict:
    """Average the mask over rings of equal distance from the DC term (0, 0)."""
    plt = _agg_backends()
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m.mean(axis=0)
    h, w = m.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(xx**2 + yy**2)
    r_norm = r / r.max()
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(r_norm, bins) - 1, 0, n_bins - 1)
    profile = np.array([m[idx == b].mean() if np.any(idx == b) else np.nan for b in range(n_bins)])
    centers = 0.5 * (bins[:-1] + bins[1:])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(centers, profile, marker="o")
    ax.set_xlabel("normalised radial frequency  (0 = DC/low, 1 = high)")
    ax.set_ylabel("mean mask weight")
    ax.set_title("Radial frequency importance")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"radius": centers, "weight": profile}


# --------------------------------------------------------------------------- --
# 3. Grad-CAM on the spatial branch
# --------------------------------------------------------------------------- --
class GradCAM:
    """Grad-CAM for the ResNet spatial branch (hooks its last conv block)."""

    def __init__(self, model) -> None:
        if model.spatial is None:
            raise ValueError("model has no spatial branch to explain")
        self.model = model
        self.target = model.spatial.backbone.layer4
        self._act = None
        self._grad = None
        self.target.register_forward_hook(self._save_act)
        self.target.register_full_backward_hook(self._save_grad)

    def _save_act(self, _m, _i, out):
        self._act = out.detach()

    def _save_grad(self, _m, _gi, go):
        self._grad = go[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        self.model.eval()
        logits = self.model(x)
        idx = logits.argmax(1) if class_idx is None else torch.full((x.size(0),), class_idx, device=x.device)
        self.model.zero_grad()
        logits.gather(1, idx.unsqueeze(1)).sum().backward()

        weights = self._grad.mean(dim=(2, 3), keepdim=True)          # GAP over spatial dims
        cam = torch.relu((weights * self._act).sum(dim=1))           # (B, h, w)
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
        return cam.cpu().numpy()


def gradcam_grid(model, images: torch.Tensor, path: Path, class_idx: int | None = None) -> np.ndarray:
    plt = _agg_backends()
    cam = GradCAM(model)(images, class_idx=class_idx)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    vis = (images.cpu() * std + mean).clamp(0, 1).permute(0, 2, 3, 1).numpy()

    n = len(images)
    fig, axes = plt.subplots(2, n, figsize=(2.4 * n, 5))
    axes = np.atleast_2d(axes)
    for i in range(n):
        axes[0, i].imshow(vis[i]); axes[0, i].axis("off")
        axes[1, i].imshow(vis[i])
        h, w = vis[i].shape[:2]
        heat = np.array(_resize(cam[i], (h, w)))
        axes[1, i].imshow(heat, cmap="jet", alpha=0.45)
        axes[1, i].axis("off")
    axes[0, 0].set_title("input", loc="left")
    axes[1, 0].set_title("Grad-CAM", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return cam


def _resize(a: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize(hw[::-1], Image.BILINEAR)) / 255.0


# --------------------------------------------------------------------------- --
# 4. feature-space projection
# --------------------------------------------------------------------------- --
@torch.no_grad()
def collect_features(model, loader, device, max_items: int = 2000):
    model.eval()
    feats, labels = [], []
    for x, y in loader:
        feats.append(model.forward_features(x.to(device)).cpu())
        labels.append(y.clone())
        if sum(f.shape[0] for f in feats) >= max_items:
            break
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()


def feature_projection(feats: np.ndarray, labels: np.ndarray, path: Path, method: str = "tsne") -> np.ndarray:
    plt = _agg_backends()
    if method == "umap":
        try:
            import umap

            emb = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(feats)
        except Exception:
            method = "tsne"
    if method == "tsne":
        from sklearn.manifold import TSNE

        emb = TSNE(n_components=2, init="pca", perplexity=min(30, len(feats) - 1), random_state=0).fit_transform(feats)

    fig, ax = plt.subplots(figsize=(6, 5))
    for lab, name, color in [(0, "real", "#4C72B0"), (1, "fake", "#C44E52")]:
        sel = labels == lab
        ax.scatter(emb[sel, 0], emb[sel, 1], s=8, alpha=0.6, label=name, c=color)
    ax.set_title(f"Fused features ({method.upper()})")
    ax.legend()
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return emb


# --------------------------------------------------------------------------- --
# CLI: regenerate every figure for one run
# --------------------------------------------------------------------------- --
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generate Task 8 figures for a trained run")
    p.add_argument("--run", required=True, help="run dir name under results/")
    p.add_argument("--data-root", required=True)
    p.add_argument("--dataset-name", default="rvf")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-root", default="results")
    p.add_argument("--projection", choices=["tsne", "umap"], default="tsne")
    p.add_argument("--n-gradcam", type=int, default=6)
    return p.parse_args(argv)


def main(argv=None):
    import torch.utils.data

    a = parse_args(argv)
    device = get_device()
    root = Path(a.results_root)
    run_dir = root / a.run
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    image_size = ckpt["args"]["image_size"]
    model = build_model(ckpt["config"], image_size=image_size, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])

    mask = model.get_frequency_mask()
    if mask is not None:
        mask_heatmap(mask, fig_dir / "mask_heatmap.png")
        radial_profile(mask, fig_dir / "radial_frequency_importance.png")
        print("  mask figures done")

    manifest = root / "manifests" / f"{a.dataset_name}_seed{a.seed}_sz{image_size}.csv"
    test_entries = read_manifest(manifest)["test"]
    ds = FaceCropDataset(test_entries, transform=_transforms(image_size, train=False))
    loader = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=2)

    if model.spatial is not None:
        imgs = torch.stack([ds[i][0] for i in range(a.n_gradcam)]).to(device)
        gradcam_grid(model, imgs, fig_dir / "gradcam.png")
        print("  grad-cam done")

    feats, labels = collect_features(model, loader, device)
    feature_projection(feats, labels, fig_dir / f"features_{a.projection}.png", method=a.projection)
    print(f"  feature projection done -> {fig_dir}")


if __name__ == "__main__":
    main()
