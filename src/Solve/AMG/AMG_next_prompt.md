# AMG — Session Prompt (integration round)

## Meta rules
- This file: lean pointer + NEXT SESSION FOCUS only, <60 lines. State lives in the integration files.
- After each session: replace NEXT SESSION FOCUS; update progress/findings/results (one-liners, numbers
  to results.json only, maintain "_index", read target entries not whole files).
- CFD/profile scripts: run_in_background=true, redirect ALL output to file, read compact summary only.
- Smoke-test scripts at 1-2 iters before full runs. Inline edits; no subagents for local file work.

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
Start U1 (mechanical move/rename per plan). P13 (F32 finest quality) FIXED 06-12: Jacobi kernels are
now residual-form x + w*Dinv*(b-Ax) — fused diag-excluded form injected ulp(|x|) noise/sweep, stalled
all-F32 CG (gf-GPU AND ref-CPU); ref-GPU was immune via cuSPARSE ext override. Validated: all-F32 F1
gf 31==ref 31; F64 45==45; 7.32ms/cycle; tests 213+194. NO open correctness items.
Uncommitted: the P13 kernel edits (reference/4, device/3, device/4) — commit first (without
Project.toml/Manifest), then begin U1. Confirm with user: fl>=1 (not fl>1) triggers matrix-free.

## Hard rules
- Refactor = behavior-preserving; the 9 plan invariants are non-negotiable.
- Don't assume, measure; every optimisation reports WHEN IT HELPS / WHEN IT DOESN'T.
- No ad-hoc knobs. GPU refresh validation must run on CUDA (CPU Atomix false-passes mixed precision).
