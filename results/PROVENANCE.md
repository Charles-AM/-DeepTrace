# Provenance notes

## The frozen tag was moved once

`results-frozen-2026-09-06` was created at commit **`a6ac339`**, then force-updated
to **`166056b`**, both on 2026-09-06.

**Why:** at `a6ac339` the ESSENCE findings table still quoted the superseded
4,000-replicate interval [−0.0351, +0.0183], and `results/canonical.json` did not
yet exist. The move brought the tag to a state where the documented numbers match
the artifacts.

**Why it should not have happened:** a freeze tag exists to identify one immutable
state. Moving it means anyone who fetched between the two pushes holds a different
"frozen" commit under the same name, which is precisely the provenance failure the
tag was created to prevent.

**Standing rule from here:** `results-frozen-2026-09-06` is immutable. Any later
result gets a **new** tag (`results-frozen-v2`, `target-sensitivity-addendum`, …).
The original continues to identify the exact state underlying the already-frozen
results.

## Superseded numbers

| quantity | superseded | current | source |
|---|---|---|---|
| crossed 95% CI | [−0.0351, +0.0183] (4,000 replicates) | **[−0.0358, +0.0180]** (50,000) | `results/canonical.json` |

The superseded interval appears deliberately in three places: the
prespecification (which quotes it as the motivation for the check and must not be
retro-edited), the crossed README's supersession note, and STABILITY.md's batch
endpoints, which are the check's own data.
