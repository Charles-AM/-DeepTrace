"""Extract face crops from video datasets (FaceForensics++, Celeb-DF v2).

FF++ and Celeb-DF ship as videos. This samples frames, detects the largest face
with MTCNN, crops it with a margin, and writes JPEGs into a ``real/`` + ``fake/``
layout that ``src.data.scan_images`` reads directly.

    python -m src.extract_faces --videos /kaggle/working/ffpp --out /kaggle/working/ffpp_crops \
        --size 160 --every 10 --max-per-video 20

Run this in a Kaggle notebook after downloading the videos, then save the (small)
crops folder as a private Kaggle Dataset for reuse.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .data import label_from_path

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def _load_detector(device: str):
    try:
        from facenet_pytorch import MTCNN
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "facenet-pytorch is required:  pip install facenet-pytorch"
        ) from e
    return MTCNN(keep_all=False, select_largest=True, post_process=False, device=device)


def _crop_with_margin(frame: np.ndarray, box, margin: float, size: int):
    from PIL import Image

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - margin * bw)); y1 = max(0, int(y1 - margin * bh))
    x2 = min(w, int(x2 + margin * bw)); y2 = min(h, int(y2 + margin * bh))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = Image.fromarray(frame[y1:y2, x1:x2]).resize((size, size), Image.BILINEAR)
    return crop


def process_video(path: Path, detector, out_dir: Path, label: int, every: int, max_per: int, size: int, margin: float) -> int:
    import cv2

    cap = cv2.VideoCapture(str(path))
    saved, fidx = 0, 0
    sub = out_dir / ("fake" if label == 1 else "real")
    sub.mkdir(parents=True, exist_ok=True)

    while saved < max_per:
        ok, frame = cap.read()
        if not ok:
            break
        if fidx % every == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, _ = detector.detect(rgb)
            if boxes is not None and len(boxes):
                crop = _crop_with_margin(rgb, boxes[0], margin, size)
                if crop is not None:
                    crop.save(sub / f"{path.stem}_{fidx:05d}.jpg", quality=95)
                    saved += 1
        fidx += 1
    cap.release()
    return saved


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Extract face crops from a video dataset")
    p.add_argument("--videos", required=True, help="root folder of videos (recursively scanned)")
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=int, default=160)
    p.add_argument("--every", type=int, default=10, help="sample every Nth frame")
    p.add_argument("--max-per-video", type=int, default=20)
    p.add_argument("--margin", type=float, default=0.3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit-videos", type=int, default=None)
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    videos = sorted(p for p in Path(a.videos).rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    if a.limit_videos:
        videos = videos[: a.limit_videos]
    if not videos:
        raise SystemExit(f"no videos under {a.videos}")

    detector = _load_detector(a.device)
    out_dir = Path(a.out)
    total = skipped = 0
    for i, vid in enumerate(videos):
        label = label_from_path(vid)
        if label is None:
            skipped += 1
            continue
        n = process_video(vid, detector, out_dir, label, a.every, a.max_per_video, a.size, a.margin)
        total += n
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(videos)} videos, {total} crops so far", flush=True)

    print(f"done: {total} crops from {len(videos) - skipped} videos ({skipped} unlabelled) -> {out_dir}")


if __name__ == "__main__":
    main()
