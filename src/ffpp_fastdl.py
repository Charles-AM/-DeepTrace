"""Parallel FaceForensics++ downloader (Phase 2 helper).

The official `download.py` fetches one file at a time, which is unusable when the
TUM server throttles per connection. This does the same thing with a thread pool
and resume support. Requires that you have already agreed to the FF++ Terms of Use
via the official request form — running this means you accept them (same as the
official script).

    python -m src.ffpp_fastdl ./ffpp --originals \
        --methods Deepfakes Face2Face FaceSwap NeuralTextures \
        --num-pairs 130 --compression c23 --server EU2 --workers 12
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SERVERS = {
    "EU": "http://canis.vc.in.tum.de:8100/",
    "EU2": "http://kaldir.vc.in.tum.de/faceforensics/",
    "CA": "http://falas.cmpt.sfu.ca:8100/",
}
METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "FaceShifter", "NeuralTextures"]


def _filelist(base_url: str) -> list[list[str]]:
    # the official script reads this from {server}v3/misc/filelist.json
    with urllib.request.urlopen(base_url + "v3/misc/filelist.json", timeout=60) as r:
        return json.loads(r.read().decode())


def _download_one(url: str, out_file: Path, timeout: int = 300) -> str:
    if out_file.exists() and out_file.stat().st_size > 0:
        return "skip"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=out_file.parent)
    os.close(fd)
    try:
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, out_file)
        return "ok"
    except Exception as e:  # noqa: BLE001
        Path(tmp).unlink(missing_ok=True)
        return f"FAIL {url.rsplit('/', 1)[-1]}: {e}"


def build_tasks(base_url, out_root: Path, methods, num_pairs, compression, originals):
    pairs = _filelist(base_url)
    if num_pairs:
        pairs = pairs[:num_pairs]
    tasks: list[tuple[str, Path]] = []

    if originals:
        ids = sorted({i for p in pairs for i in p})
        for vid in ids:
            u = f"{base_url}v3/original_sequences/youtube/{compression}/videos/{vid}.mp4"
            tasks.append((u, out_root / "original_sequences/youtube" / compression / "videos" / f"{vid}.mp4"))

    for m in methods:
        for a, b in pairs:
            for name in (f"{a}_{b}", f"{b}_{a}"):
                u = f"{base_url}v3/manipulated_sequences/{m}/{compression}/videos/{name}.mp4"
                tasks.append((u, out_root / "manipulated_sequences" / m / compression / "videos" / f"{name}.mp4"))
    return tasks


def main(argv=None):
    p = argparse.ArgumentParser(description="Parallel FaceForensics++ downloader")
    p.add_argument("output_path", type=str)
    p.add_argument("--methods", nargs="+", default=["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"], choices=METHODS)
    p.add_argument("--num-pairs", type=int, default=130, help="first N video pairs (0 = all ~1000)")
    p.add_argument("--compression", default="c23", choices=["raw", "c23", "c40"])
    p.add_argument("--server", default="EU2", choices=list(SERVERS))
    p.add_argument("--originals", action="store_true", help="also download the pristine videos")
    p.add_argument("--workers", type=int, default=12)
    a = p.parse_args(argv)

    base_url = SERVERS[a.server]
    out_root = Path(a.output_path)
    tasks = build_tasks(base_url, out_root, a.methods, a.num_pairs, a.compression, a.originals)
    print(f"{len(tasks)} files -> {out_root}  ({a.workers} workers, {a.server}, {a.compression})", flush=True)

    done = skipped = failed = 0
    fails: list[str] = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_download_one, u, f): u for u, f in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r == "ok":
                done += 1
            elif r == "skip":
                skipped += 1
            else:
                failed += 1
                fails.append(r)
            if i % 25 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}  ok={done} skip={skipped} fail={failed}", flush=True)

    if fails:
        print("\nFAILURES (re-run to retry — completed files are skipped):")
        for f in fails[:40]:
            print("  " + f)
    print(f"\ndone: {done} downloaded, {skipped} already present, {failed} failed")


if __name__ == "__main__":
    main()
