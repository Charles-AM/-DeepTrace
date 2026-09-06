"""FF++ crop-filename parsing — the single source of truth.

Lives apart from `src/predict.py` because that module imports torch, and the
split-construction and analysis tools that need this parsing must run on a CPU
box with no deep-learning stack. Duplicating the regexes instead would be worse:
the Face2Face digit bug (see below) was a one-character mistake that silently
emptied 600 rows, and it must only ever be possible to make it in one place.
"""

from __future__ import annotations

import re
from pathlib import Path

# manipulated: .../manipulated-sequences-<Method>-<comp>-videos-<target>-<source>_<frame>.jpg
# original:    .../original-sequences-youtube-<comp>-videos-<id>_<frame>.jpg
# NOTE: the method group must allow digits — "Face2Face" contains one, and an
# [A-Za-z]+ class silently failed to match it, leaving those rows with an empty
# video_id that would have collapsed into a single bogus cluster. Caught by the
# unparsed-rows gate in the V1 pipeline; see tests/test_clusters.py.
_FAKE_RE = re.compile(
    r"manipulated-sequences-(?P<method>[A-Za-z0-9]+)-(?P<comp>c\d+|raw)-videos-"
    r"(?P<target>\d+)-(?P<source>\d+)_(?P<frame>\d+)"
)
_REAL_RE = re.compile(
    r"original-sequences-\w+-(?P<comp>c\d+|raw)-videos-(?P<target>\d+)_(?P<frame>\d+)"
)


def parse_crop_name(path: str) -> dict:
    """Metadata from a crop filename. `video_id` is the grouping key used by
    `--group-by 'videos-([0-9]+)'`; `source_seq` is populated for fakes only and is
    what component-level clustering needs (see src/clusters.py)."""
    name = Path(path).name
    m = _FAKE_RE.search(name)
    if m:
        d = m.groupdict()
        return {"manipulation": d["method"], "compression": d["comp"],
                "target_seq": d["target"], "source_seq": d["source"],
                "video_id": d["target"], "frame_idx": int(d["frame"])}
    m = _REAL_RE.search(name)
    if m:
        d = m.groupdict()
        return {"manipulation": "real", "compression": d["comp"],
                "target_seq": d["target"], "source_seq": "",
                "video_id": d["target"], "frame_idx": int(d["frame"])}
    return {"manipulation": "?", "compression": "?", "target_seq": "",
            "source_seq": "", "video_id": "", "frame_idx": -1}



__all__ = ["parse_crop_name", "_FAKE_RE", "_REAL_RE"]
