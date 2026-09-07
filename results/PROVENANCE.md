# Provenance notes

## Tags — all immutable from here

| tag | commit | state |
|---|---|---|
| `results-frozen-2026-09-06-orig` | `a6ac339` | the state this name **first** pointed at. ⚠️ ESSENCE §8 still quoted the superseded 4,000-replicate interval here, and `results/canonical.json` did not exist. |
| `results-frozen-2026-09-06` | `166056b` | corrected: canonical block present, stale interval removed. **This is the frozen core-results state.** |
| `results-frozen-v2` | `d6af5c9` | adds the prespecified target-group sensitivity analysis. Core results identical. |

### The move, recorded

`results-frozen-2026-09-06` was created at `a6ac339`, then force-updated to
`166056b`. A freeze tag exists to identify one immutable state; moving it means
anyone who fetched between the two pushes holds a different commit under the same
name — precisely the provenance failure the tag was meant to prevent.

It is **not** being moved a third time. Both states are addressable by the tags
above, and no tag name changes meaning from here.

## Prespecification history — three distinct statements

Stating this as one sweeping "before the runs" would be less credible than the
truth, which differs by arm:

1. **Video-aggregated stability rule** (`86e6c98`) — specified *after* inspecting
   the preliminary 4,000-replicate result, and *before* the confirmatory
   high-replicate runs. Analysis governance, not preregistration before any
   result.
2. **Frame-pooled rule** (same commit) — specified *before any frame-pooled
   crossed result existed*. This is the arm where the rule actually bound,
   returning a borderline verdict rather than a forced one.
3. **Target-group sensitivity** (`0639ba0`) — specified *before the analysis was
   implemented*; `cluster_boot` and `crossed_boot` offered only
   frame/video/component at that commit.

## Superseded numbers

| quantity | superseded | current | source |
|---|---|---|---|
| crossed 95% CI | [−0.0351, +0.0183] (4,000 replicates) | **[−0.0358, +0.0180]** (50,000) | `results/canonical.json` |

The superseded interval appears deliberately in three places: the crossed
prespecification (which quotes it as the motivation and must not be retro-edited),
the crossed README's supersession note, and STABILITY.md's batch endpoints, which
are the check's own data.
