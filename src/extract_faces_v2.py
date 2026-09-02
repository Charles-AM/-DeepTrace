"""Face-crop extractor v2 — crops are named by a slug of the video's path so the
FF++ manipulation methods (which reuse pair filenames) can't overwrite each other.
Supersedes extract_faces.py; --video-list filters to an official test split.

    python -m src.extract_faces_v2 --videos <dir> --out <dir> [--video-list list.txt]
"""
from __future__ import annotations
import argparse, re
from pathlib import Path

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
FAKE = ("fake", "manipulated", "deepfake", "synthesis", "faceswap", "neuraltextures", "face2face")
REAL = ("real", "original", "pristine", "genuine", "youtube")


def label_from_path(p) -> int | None:
    s = str(p).lower()
    if any(k in s for k in FAKE):
        return 1
    if any(k in s for k in REAL):
        return 0
    return None


def _crop(frame, box, margin, size):
    from PIL import Image
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - margin * bw)); y1 = max(0, int(y1 - margin * bh))
    x2 = min(w, int(x2 + margin * bw)); y2 = min(h, int(y2 + margin * bh))
    if x2 <= x1 or y2 <= y1:
        return None
    return Image.fromarray(frame[y1:y2, x1:x2]).resize((size, size), Image.BILINEAR)


def process_video(path: Path, root: Path, det, out_dir: Path, label: int,
                  every: int, max_per: int, size: int, margin: float) -> int:
    import cv2
    slug = re.sub(r"[^0-9A-Za-z]+", "-", str(path.relative_to(root).with_suffix("")))
    sub = out_dir / ("fake" if label == 1 else "real")
    sub.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(path))
    saved = fidx = 0
    while saved < max_per:
        ok, frame = cap.read()
        if not ok:
            break
        if fidx % every == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, _ = det.detect(rgb)
            if boxes is not None and len(boxes):
                c = _crop(rgb, boxes[0], margin, size)
                if c is not None:
                    c.save(sub / f"{slug}_{fidx:05d}.jpg", quality=95)
                    saved += 1
        fidx += 1
    cap.release()
    return saved


def _read_list(path: Path) -> set[str]:
    out = set()
    for ln in Path(path).read_text().splitlines():
        ln = ln.strip()
        if ln:
            out.add(ln.split()[-1].replace("\\", "/"))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Face-crop extractor v2")
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--every", type=int, default=12)
    ap.add_argument("--max-per-video", type=int, default=20)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--video-list", default=None)
    ap.add_argument("--limit-videos", type=int, default=None)
    a = ap.parse_args(argv)

    from facenet_pytorch import MTCNN
    det = MTCNN(keep_all=False, select_largest=True, post_process=False, device=a.device)
    root = Path(a.videos)
    vids = sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    if a.video_list:
        want = _read_list(Path(a.video_list))
        vids = [p for p in vids if str(p.relative_to(root)) in want or p.name in want]
        print(f"video-list matched {len(vids)} / {len(want)}")
    if a.limit_videos:
        vids = vids[: a.limit_videos]

    tot = skip = 0
    for i, v in enumerate(vids):
        lab = label_from_path(v)
        if lab is None:
            skip += 1
            continue
        tot += process_video(v, root, det, Path(a.out), lab, a.every, a.max_per_video, a.size, a.margin)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(vids)} videos, {tot} crops", flush=True)
    print(f"done: {tot} crops from {len(vids)-skip} videos ({skip} unlabelled) -> {a.out}")


if __name__ == "__main__":
    main()
