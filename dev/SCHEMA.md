# Context Vault Manifest (`dev/`)

| File Path | Description | Budget | Access Pattern |
| :--- | :--- | :--- | :--- |
| `dev/SCHEMA.md` | Vault layout and budget exceptions | <= 50 lines | Read / Amend |
| `dev/activeContext.md` | Live session state and resumption | <= 60 lines | Read / Write |
| `dev/roadmap.md` | Phases `P<n>`, their order and status | <= 120 lines | Read / Write |
| `dev/phaseRoadmap.md` | The active phase: milestones `P<n>-M<k>` and exit gate | <= 120 lines | Read / Write |
| `dev/plans/p<n>-m<k>-<slug>.md` | Live milestone plans, steps `P<n>-M<k>-S<j>` | <= 150 lines each | Read / Write |
| `dev/spec.md` | What the product must be true of | <= 120 lines | Read / Amend |
| `dev/architecture.md` | Current structure and mechanisms | <= 100 lines | Read / Update |
| `dev/gotchas.md` | Traps, constraints, how to work here | <= 90 lines | Read / Update |
| `dev/decisions.md` | Decisions and refusals | <= 3 lines per entry | Append-only |
| `dev/telemetry/*` | Benchmarks and gate results | Unlimited | Append-only |
| `dev/scripts/*` | Indexed project-specific helpers | Unlimited | Read / Update |
| `dev/features/<slug>/` | One feature's own vault, same records | Same budgets | Read / Write |
| `dev/archive/*` | Historical, non-authoritative records | Unlimited | Read-only |

IDs are allocated and never renumbered: position is order, the ID is identity. A milestone or step closed without delivering keeps its line and names where the work went.

Only one `phaseRoadmap.md` is live. At phase close it moves to `dev/archive/phases/<phase>/roadmap.md`, then resets for the next phase or records no active phase. Each live milestone plan is linked from that roadmap and moves, under the same name, to `dev/archive/plans/<phase>/` when its milestone closes.

One block is one line: a paragraph, list item, table row, requirement or decision entry is never hard-wrapped, and no block exceeds 2,600 characters (a requirement 480, a decision entry 800). A line budget therefore counts BLOCKS.

## Records that constrain content, not length

`dev/spec.md` holds only clauses a completely different implementation could satisfy; mechanisms, names and procedure belong to `dev/architecture.md`, `dev/gotchas.md` and `dev/decisions.md`. `dev/decisions.md` holds one entry per decision of at most three lines, citing evidence rather than inlining it.

## Budget exceptions

None. Use `<path>: <limit> - <reason>` only when preserving binding content requires it.

`check`'s repository-wide comment rule reports twelve pre-existing blocks of commented-out post-processing code in shipped `examples/*.jl` that predate this vault and belong to no phase here. They are commented-out code, not narrative, and rewriting shipped examples is outside P1's scope: read `check` with `grep -v '^- comments:'` while they stand.
