"""Identity clustering over the FaceForensics++ pair graph.

FF++ manipulated videos are named ``<target>_<source>``: the video pairs two
sequences. Grouping only on the target (what ``--group-by 'videos-([0-9]+)'``
does) therefore still lets an identity cross a split boundary — sequence 982 can
be the *source* of `004_982` and the *target* of `982_004`, and those land in
different groups. CADDM (CVPR 2023) shows detectors latch onto identity as a
shortcut, so this residual overlap is a real leak, not a technicality.

The fix: treat each sequence id as a graph node and each manipulated video as an
edge between its target and source. Connected components are then identity
clusters that share no sequence, and assigning whole components to a split
guarantees no sequence appears on both sides.

Used for (a) identity-disjoint splits and (b) the resampling unit in
`src/cluster_boot.py` — resampling individual videos is optimistic when several
manipulations share a target.
"""

from __future__ import annotations

from collections import defaultdict


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_identity_clusters(pairs: list[tuple[str, str]]) -> dict[str, str]:
    """``pairs`` = (target_seq, source_seq) for every manipulated video; source may
    be "" for originals. Returns ``sequence_id -> component_id`` (component id is
    the lexicographically smallest sequence in the component, so it is stable
    across runs)."""
    uf = _UnionFind()
    for target, source in pairs:
        if not target:
            continue
        uf.find(target)
        if source:
            uf.union(target, source)

    members: dict[str, list[str]] = defaultdict(list)
    for node in uf.parent:
        members[uf.find(node)].append(node)
    out: dict[str, str] = {}
    for root, nodes in members.items():
        cid = min(nodes)
        for n in nodes:
            out[n] = cid
    return out


def cluster_stats(clusters: dict[str, str]) -> dict:
    sizes: dict[str, int] = defaultdict(int)
    for cid in clusters.values():
        sizes[cid] += 1
    counts = sorted(sizes.values(), reverse=True)
    return {"n_sequences": len(clusters), "n_components": len(sizes),
            "largest": counts[0] if counts else 0,
            "singletons": sum(1 for c in counts if c == 1),
            "size_distribution": counts[:10]}


def clusters_from_prediction_rows(rows: list[dict]) -> dict[str, str]:
    """Convenience: build clusters directly from `src.predict` output rows."""
    pairs = {(r.get("target_seq", ""), r.get("source_seq", "")) for r in rows}
    return build_identity_clusters(sorted(pairs))
