# Handling rule — the accidental 2026-09-18 full re-run

**Written while the run is still executing**, before its C1 and C2 results are
known. Git history is the evidence. Deciding how to treat data after seeing it is
the failure this project is about.

## What happened

The re-run was intended as `--stages B1,B2`. The flag did not reach the run cell,
so the driver executed **all four stages**. B2 completed correctly at ~40 minutes
(`train=23250 test=750`, manifest guard passed). C1 and C2 then re-ran at
**identical settings** in a **different session** from 2026-09-17.

No result from the re-run's C1 or C2 was inspected before this file was committed.
One figure was visible in the streaming log and is recorded below, because
pretending otherwise would be false.

| visible before writing this rule | value |
|---|---|
| `c2_rep1_xception_seed0` test AUC, re-run | 0.7949 (2026-09-17: 0.8043) |

## C1 — replication is PERMITTED

`docs/c1-lr-prespecification.md` §6 prohibits four things: arm-selective lr
selection, **extending** the epoch budget, adding lr values, and re-running the
frozen headline. **A straight replication at identical settings is none of these.**

**Rule:** the re-run sweep is reported as an independent replication at the same
settings. It is informative on two prespecified points:

- whether the **lr 1e-3 divergence reproduces** (Xception reached exactly chance
  with a frozen training loss). If it does, divergence at that rate is a property
  of the configuration, not a one-off failure.
- whether the **epoch-4 peaks reproduce**, which bears on whether the 5-epoch
  budget was systematically too tight or those two cells were unlucky.

⚠️ The two sweeps are **not pooled**. Each cell remains n=1 per sweep. Agreement
between them is reassurance, not additional precision.

## C2 — NOT pooled, reported as declared replication

`docs/c2-repeat-prespecification.md` §6 prohibits *"adding further repeats because
the first two looked interesting."* The motive here was an omitted CLI flag, not
the result — **but motive is not verifiable by a reader, only the data is.**

**Rule, fixed now:**

1. The re-run repeats are **never pooled** with the 2026-09-17 repeats. The
   sd 0.0226 figure across three nominally identical runs **stands as reported**
   and is not recomputed with more observations.
2. They are reported as an **accidental cross-session replication**, described as
   accidental, with this file cited as evidence the rule predates the data.
3. The permitted use is the **cross-session comparison** the original
   prespecification wanted and could not make: the same nominal configuration run
   in two different sessions. That is what §4 branch (b) — "a session-level
   component is indicated" — was written to test.
4. **All §5 limits still bind**, in particular: **no causal attribution**, under any
   outcome, however systematic the pattern appears.

## What may NOT be done with either

- Reporting whichever sweep or repeat set looks tidier
- Presenting the pooled spread as though it were prespecified
- Revising the sd 0.0226 figure upward or downward using the new runs
- Treating an accidental replication as though it had been planned

## If the two sets disagree materially

That is a **finding about repeatability**, not a problem to resolve by choosing.
Report both, with this file as the record that the handling rule was fixed before
the data existed.
