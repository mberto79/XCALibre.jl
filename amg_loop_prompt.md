# AMG Optimization Agent

You are running headlessly inside the XCALibre.jl project directory. Your mission is to
optimize the AMG preconditioner to achieve a runtime ratio **< 0.60x** vs Cg+Jacobi baseline
on the F1 1.67M-cell RANS KOmega CUDA GPU benchmark.

## Orientation

Read these two files FIRST before doing anything else:

1. `amg_loop_state.md` — full history, closed approaches, current config, what to try next
2. `F1-fetchCFD_Minimal/amg_loop_results.txt` — latest benchmark output (timings, PCG iters, phase breakdown)

The AMG source lives in `src/Solve/AMG/` (files AMG_0_types.jl through AMG_6_api.jl) and
`ext/XCALibre_CUDAExt.jl`. The benchmark config section is clearly marked `[AMG Config]` in
`F1-fetchCFD_Minimal/amg_loop_profile.jl`.

Also read `src/AMG_changes_and_optimisation_after_profiling.md` for the full closed-steps log
so you don't repeat closed approaches. However, the last 4 approaches may have been corrupted and could be explored again.

## Your Task (execute in this order)

1. Read `amg_loop_state.md` and `F1-fetchCFD_Minimal/amg_loop_results.txt`
2. Read `src/AMG_changes_and_optimisation_after_profiling.md` (closed approaches list)
3. Identify the single highest-leverage change based on the phase breakdown in results
4. Call `advisor()` BEFORE implementing any non-trivial algorithmic change (GPU kernels,
   new data structures, coarsest-level solver changes)
5. Make ONE targeted change — either:
   - Modify AMG source in `src/Solve/AMG/` or `ext/XCALibre_CUDAExt.jl` (algorithmic), OR
   - Adjust parameters in the `[AMG Config]` section of `F1-fetchCFD_Minimal/amg_loop_profile.jl`
6. Update `amg_loop_state.md`:
   - Increment "Current Iteration" number
   - Record what you changed and why (1-2 lines)
   - Append a row to the History table with ratio from the LAST run (not this one)
   - Update "What to Try Next" with your rationale for next iteration
   - If no further paths exist, write `STATUS: EXHAUSTED` (the loop will stop)

## Hard Constraints

- **Do NOT run full julia benchmark, only smoke tests with 2 iterations if you need to check implementation changes work** — the orchestration script handles simulation
- **Do NOT modify** test files, docs, non-AMG solver files, or examples
- **Preserve** the `BENCHMARK_*` output lines and variable names in `amg_loop_profile.jl`
- **Keep** `amg_loop_state.md` under 150 lines (compress old detail ruthlessly)
- **One change per iteration** — resist the urge to make multiple changes at once
- Follow the coding conventions in CLAUDE.md (succinct comments, no narration, `!` suffix for mutating)

## Context You Already Have

CLAUDE.md is auto-loaded (full project guide). Your project memory files are also loaded.
You do NOT need to re-read CLAUDE.md — just follow its conventions.

The `advisor()` tool is available. Use it before implementing GPU kernel changes or new
data structures — it has seen the full conversation history.
