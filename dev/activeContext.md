# Active context - distributed module release polish
LOAD: dev/activeContext.md, dev/spec.md, dev/gotchas.md, dev/architecture.md, dev/roadmap.md, dev/phaseRoadmap.md
updated: 2026-09-17T00:00:00+01:00
STATE: BUILDING
STEP: P1-M1-S1 - reconcile the development record and track it
HEAD: 9ca6f2c2
BRANCH: HM/distributed-draft
GATE: xcalibre-dev check .
resume: remove both `dev/` entries from `.gitignore`, add the machine-specific exclusions, run `xcalibre-dev check`, then commit and push the vault together with the backward-facing-step example fix
## position
P1-M1 is in flight. The vault is scaffolded and filled; `.gitignore` and the commit remain. Nothing else in the phase has started.
## evidence
- The previous round's findings, including the scaling table and the environment traps, are carried into `dev/gotchas.md` and this phase's milestones; `POLISH_prev_findings.md` is superseded once P1-M1 lands (D2).
- The backward-facing-step example's hang was ranks dispatching differently on a value only rank zero held; the fix is in the working tree and P1-M4 makes that bug class unrepresentable.
## blocked/carried
- Whether the AMG device coarse-solve environment variables in `src/Solve/AMG/` belong to this phase is the user's call; they arrived with a different feature.
