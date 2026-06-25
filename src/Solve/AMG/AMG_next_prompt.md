# AMG — Session Prompt (integration round)

## Meta rules
- This file: lean pointer + NEXT SESSION FOCUS only, <60 lines. State lives in the integration files.
- After each session: replace NEXT SESSION FOCUS; update progress/findings/results (one-liners, numbers
  to results.json only, maintain "_index", read target entries not whole files).
- CFD/profile scripts: run_in_background=true, redirect ALL output to file, read compact summary only.
- Smoke-test scripts at 1-2 iters before full runs. Inline edits; no subagents for local file work.
- DO NOT COMMIT. STAGE ONLY

## Files (this round)
AMG_integration_plan.md       — THE plan (layout, renames, dispatch, tests, gates). Read fully once.
AMG_integration_progress.json — U1-U4 status
AMG_integration_results.json  — baseline oracle + harness snippet (NEVER delete entries)
AMG_integration_findings.md   — decisions/gotchas only
Archived/                     — all prior rounds (perf P1-P13, arch). Consult only when a number/why is needed.

## Delegation (token policy)
- U1 and U3 are mechanical: spawn ONE Sonnet subagent per phase, prompt = the relevant plan sections
  verbatim + acceptance gate + "no logic changes". Main model reviews diff + runs gates.
- U2 (struct/dispatch) and anything kernel-adjacent: main model. Never Haiku for kernels.
- Gates after every phase: both test files + standalone F1 numbers vs results.baseline.

## Session setup
Pkg.activate("/home/humberto/Julia/XCALibre.jl"); then Pkg.add("CUDA") — LOCAL-ONLY, git restore
Project.toml/Manifest.toml before staging. S = XCALibre.Solve. F1 fixture + harness: results.json "harness".
Struct change -> REPL RESTART. MCP run-julia-code returns final value only. ONE Julia process (14GB box).

## NEXT SESSION FOCUS
U1-U4 + messaging + acceptance + comment sweep ALL DONE. ALL staged, NOT committed (stage-only holds).
06-25: F1 acceptance re-run with USER-EDITED case settings (ml3/mc124/pre3post3/F32coarse/F64sys) -> mf==ref
16 iters, trueres 8.25e-4, PASS (results.json U4_accept). Comment sweep of files 0-6/8/9 done (Sonnet,
-307/+76, comment-only, precompile OK); 4 over-cut precision/invariant gotchas restored by hand (findings
## Acceptance + comment sweep). U4 tests: T1/T2 in test_AMG_matrices, T3 in test_AMG CUDA block, gate green.
REMAINING: only (1) COMMIT when user approves (currently stage-only). Optionally re-run CPU test files after
the comment sweep if extra-cautious (sweep was comment-only + precompile OK, so unlikely needed).
GATE: both test files (0 fail) + F1 CUDA harness (results.json U3_gate 45/45, 31/31; U4_accept 16/16 case-settings).

## Hard rules
- Refactor = behavior-preserving; the 9 plan invariants are non-negotiable.
- Don't assume, measure; every optimisation reports WHEN IT HELPS / WHEN IT DOESN'T.
- No ad-hoc knobs. GPU refresh validation must run on CUDA (CPU Atomix false-passes mixed precision).
