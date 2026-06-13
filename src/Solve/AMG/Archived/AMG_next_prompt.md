# AMG — Session Prompt

## Meta rules (enforce every session)
- This file: lean pointer + NEXT SESSION FOCUS only. Keep under 80 lines total.
- Do NOT record state here. State lives in progress/findings/results files.
- After each session: replace NEXT SESSION FOCUS (don't append history). Update perf files only.
- Token: read files by key/section — never whole-file. Run CFD scripts background+redirect.
  Inline code edits. Spawn subagents only for web or large parallel exploration or when a task is suitable for a cheaper model.
- JSON files: always maintain "_index" key at top (key -> one-liner); read target entry not whole file.

## Archive (implementation phase)
src/Solve/AMG/Implementation/
  arch_findings.md   — decisions/gotchas/KA pitfalls
  arch_plan.md       — phase list + key structs
  arch_progress.json — final impl state + test counts
  arch_results.json  — all impl benchmark numbers (oracle; NEVER delete entries)

## Performance-round files (create at start of perf session if absent)
  AMG_perf_progress.json  — {"_index":{..},"tasks":[{"id":"P1","desc":"..","done":false,"note":".."}]}
  AMG_perf_findings.md    — ## section headers; one-liner facts; no duplication with arch_findings
  AMG_perf_results.json   — {"_index":{..}, "result_key":{numbers+1-line-caveat}}

## Session setup
```
Pkg.activate("/home/humberto/Julia/XCALibre.jl"); using XCALibre, CUDA
# CUDA = LOCAL-ONLY dep — DO NOT COMMIT Project.toml/Manifest.toml (git restore before commit)
S = XCALibre.Solve
# F1 fixture:
using Serialization; e=open(deserialize,"/home/humberto/casesXCALibre/F1-fetchCFD_Minimal/f1_pressure_systems.bin")[1]
```
- AMG struct change -> REPL RESTART required (Revise can't handle struct redefinition)
- MCP run-julia-code returns final expression value only (no stdout/stderr)
- All CFD/profile scripts: run_in_background=true; redirect all output to file; read compact summary
- Profile tools: NVTX.jl ranges, CUDA.@profile

## NEXT SESSION FOCUS
P9-P12 DONE 06-10/11 — ALL FOUR directive levers closed; **AMG now beats PCG+Jacobi** (results.P12_*).
F1 SIMPLE 50-iter same-run: gf ml3 interval4 rtolp1e-1 **0.6349 s/it vs jacobi 0.6522 (-2.7%)** at
matched p/Ux quality; rtolp1e-2 +4.4% with 3.6x deeper p. Static pressure: 45it/0.44s vs 1078it/1.44s
(3.2x tight), 2.3x loose. Honest claim: loose-tol steady = parity-to-faster (margins ~ run variance);
decisive at tight tol / transient / VRAM. Tests 213/213 + 194/194.

WHAT LANDED (stage: device/4,5,7 + reference/5,6 + test_AMG.jl + test_AMG_matrices.jl):
- P9 Lever A: gf standalone 151->121==ref. Reference _cycle! ALIASES level.rhs at l>=2 (post-smooths
  vs pre-smoothing residual, stronger than textbook). mf_ml_cycle + host oracle replicate it.
- P10 Lever C: sc-in-Cg via flexible PR+ beta (_amg_cg_flexible, r_prev=workspace.correction). 89->45
  iters. sc DEFAULT-ON in Cg now for ref AND gf (user-facing change). mf sc reuses down-sweep lv.r.
- P11 Levers B+D: fuse_levels>1 transient-refreshable via composed maps (comp_dst/comp_s in plan);
  fused Gershgorin = exact |diag| + cross-aggregate abs-sum (naive abs-sum -> omega 20-40% off).
  Warm-start lambda (plan.eig, warm 2 iters): refresh 27->20-24ms. VERDICT fuse>1: 1.5-1.9x SLOWER
  solve, -33MB only -> fuse_levels=1 stays deploy (directive latency premise refuted by measurement).
- P12: gf coarse_refresh_interval support (light finest-only refresh off-iterations, mirrors reference;
  refresh_mf_ml! coarse=false). Deploy config: ml3 fuse1 F64 mc1024 V(2,2) sc interval<=4 rtolp 1e-1/1e-2.

OPEN / next (lower priority):
- F32 finest cycle quality (p3_root_cause): research-grade; deploy stays finest-F64 + coarse F32.
- No open correctness items. AMGSolver-standalone gap CLOSED (P9).
- OPS: ONE Julia process at a time (14GB box). Device matrix via ModelFramework._build_A(backend,I,J,V,n)
  (findnz(parent(A))) — bare CuSparseMatrixCSR, faithful to live CFD type. REPL can run CPU test files
  via include (needs `using SparseMatricesCSR` first — runtests-only dep).
- Project.toml LOCAL-ONLY (CUDA): restored this session; re-Pkg.add for REPL work, restore before staging.
- Bench scripts: casesXCALibre/F1-fetchCFD_Minimal/f1_amg_{vs_jacobi,tol_sweep,interval}.jl (50-iter SIMPLE).

## Hard rules
- GREENFIELD: don't touch reference/; device/ owns its pipeline
- Don't assume, measure, test, validate.
- No adhoc solutions or knobs to circumvent a solution. Tuning only once implementation has reached optimum performance/architechture
- Don't delegate kernel work to Haiku (KA pitfalls critical; use Opus/Sonnet)
- After each task: update perf progress + findings + results files
- Report WHEN IT HELPS / WHEN IT DOESN'T for every optimisation
- GPU: RTX 4070 8GB; always watch VRAM

## Token efficiency lessons (self-update each session)
- run_in_background=true + redirect for all CFD; read only compact summary file
- JSON: "_index" first; read target entry not whole file
- Inline code edits; no subagent for local file work
