# P1-M26 — shipped precompilation of the solver path (plan)

Linked from `dev/phaseRoadmap.md`. Requirements: R11, R4. Governing decisions: D106, D107, D108.

## Problem, quantified

First-run compilation drives per-rank private memory: a per-case precompile recipe cuts private at `iterations` 559 → 359 MB and compile 12.1 → 0.04 s at 10 mm n=4 (D106, D108), but 256 of 937 harvested statements bake a rank-local kernel size, so nothing can ship as-is, and case-independent signatures alone gain nothing (D107). 188 kernel constructions build `f!(_setup(backend, workgroup, n)...)`, which makes the launch size (and, under `AutoTune`, the workgroup) a type parameter: every distinct mesh, patch and rank size compiles its own kernel.

## Approach

(1) Size-free launches: one helper builds each kernel without its range and passes the range at launch, so a compiled kernel serves any mesh and rank count; the call sites change mechanically; an integer workgroup stays static because a dynamic one cost 1.6 percent CPU time, and only `AutoTune` passes its workgroup at launch (amended by D133). (2) A PrecompileTools workload over the solver call tree for covered cases, sized after measuring what it adds to package precompilation, since every user pays that (R4).

## Steps

- [x] **P1-M26-S1** DELIVERED (D133): size-free kernel launches through one helper at every `_setup` construction site — verdict: residual hashes bitwise equal at 10 mm n=2,4 (`mem_probe.jl`), CPU per-iteration time within 3 percent, gate green, `test_gpu.jl` n=1 green; recorded: compile time of a second mesh size in the same session before and after.
- [-] **P1-M26-S2** WITHDRAWN (D134, user): measure the workload options: serial solver cases only, and serial plus the single-rank distributed path with the PETSc extension; for each, package precompile time added, load time, and first-run compile and private memory at 10 mm n=4 for a covered case — verdict: numbers recorded and the scope chosen by the user.
- [-] **P1-M26-S3** WITHDRAWN (D134, user): the chosen workload — verdict: first-run private memory and compile time at 10 mm n=4 within 10 percent of the per-case recipe for a covered case, per-iteration time within 3 percent, residuals bitwise identical, gate green.
- [x] **P1-M26-S4** DELIVERED (D135): the guide's precompile recipe says what now ships and when the recipe still helps — verdict: docs build green.

## Exit criterion

Amended by D134: S1 landed and S4 documents what it changes for the recipe; the workload bars no longer apply. Was: the roadmap row's exit: first-run private memory and compile time at 10 mm n=4 within 10 percent of the per-case recipe for a covered case, per-iteration time within 3 percent on CPU and GPU, residuals bitwise identical, package precompile and load time recorded, gate green.

## Open questions

- SETTLED (D134): no workload ships.
- Not pursued: PETSc initialisation inside a precompile workload (S2 withdrawn).
