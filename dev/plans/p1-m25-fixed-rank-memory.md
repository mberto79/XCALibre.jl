# P1-M25 - per-rank fixed memory (plan)

Linked from `dev/phaseRoadmap.md`. Requirements: R11 (and R4 for the serial load time). Governing decisions: D90, D96, D99, D100, D101.

## Problem, quantified

Every rank holds about 790 MB before it owns a cell (fit over 10/5/4 mm n=2, `dev/telemetry/memory_breakdown.md`): 546 MB RSS after `using XCALibre, PETSc, MPI` and `MPI.Init`, with only 23 MB of it live Julia heap, plus about 260 MB that appears over the first run (compiled code and allocator slack; the 10 mm residual). Below about 440k cells per rank this exceeds the whole per-cell cost, and eight ranks on one node pay it eight times. RSS counts shared file-backed pages (system image, package images, `libpetsc`, `libLLVM`) in full in every process although the node holds them once, so the private per-rank cost is not yet known.

## Approach

Measure before curing. First split RSS into shared and private pages per mapping (`/proc/self/smaps`), then attribute the private part to loaded packages and to runtime compilation. Cures are chosen from that attribution; none is prescribed here. Candidates to measure, not to assume: runtime compilation of the distributed SIMPLE path that a precompile workload would move into shared package-image pages; PETSc.jl loading every configured PETSc library rather than the one selected; dependencies loaded but unused on a CPU run; a system image. Each cure is one step with its own bar.

## Configuration space

env {`dev/petscenv_stock`, `dev/petscenv_conda_ompi`} x ranks {1, 2, 4} x mesh {10 mm} x iterations {2}, measured {RSS, PSS, USS} at each `mem_probe.jl` stage. 10 mm keeps every run under the five-minute cap (D101) and the fixed cost is 65 percent of its peak. A result is confirmed once at 5 mm n=2 at milestone close.

## Steps

- [x] **P1-M25-S1** DELIVERED (D102): 248 MB shared per node + 324 private per rank at runtime (n=4), growth over the run is GC heap and malloc not JIT code; see `dev/telemetry/fixed_rank_memory.md`. Was: `mem_probe.jl` reads `/proc/self/smaps_rollup` (Rss, Pss, Private_Clean, Private_Dirty, Shared_Clean, Shared_Dirty) at every stage and, at `runtime` and `iterations`, the top 25 mappings of `/proc/self/smaps` by private bytes - mechanism: measurement - cost: two runs (n=1, n=4) at 10 mm, each under five minutes - verdict: a table in `dev/telemetry/fixed_rank_memory.md` splitting the fixed cost into shared per node and private per rank, attributed to mappings (anonymous heap, JIT code, package images, `libpetsc`, `libLLVM`, CUDA libraries).
- [x] **P1-M25-S2** DELIVERED (D103, D104): ranked rows are GC page retention about 165 MB and PETSc.jl 47 MB private per rank at n=4; see telemetry § S2. Was: loaded-package audit: modules and shared libraries present in a CPU worker after setup, and the private bytes added by `using` each heavy dependency alone in a fresh process - mechanism: measurement - cost: one script under five minutes - verdict: a ranked list of private MB per dependency in the same telemetry file; each cure step below names its row.
- [ ] **P1-M25-S3** cure 1, chosen from S1/S2 and written here before it is built - mechanism: <from S1/S2> - cost: <load time, precompile time> - verdict: private fixed MB per rank at 10 mm n=4 falls by the attributed amount; serial load time within 10 percent; residual hashes unchanged.
- [ ] **P1-M25-S4** cure 2 or WITHDRAWN if S3 leaves nothing above 10 percent of the fixed cost - same bar.
- [ ] **P1-M25-S5** the per-node memory recipe in the distributed guide states the measured private and shared cost, so a user can size ranks per node - mechanism: documentation - cost: none - verdict: docs build green.

## Exit criterion

Baseline (pinned by D104): private MB at `iterations`, rank 0, 10 mm n=4, `gc=0`, `dev/petscenv_stock` = 556 (RSS 817), residual hash `dbc3c69ab48b3394`; 30 percent means 389 or less. Raw: `dev/telemetry/memory_breakdown/m25s1_10mm_n4.txt`.


The fixed per-rank cost is split into private and shared bytes and attributed to named mappings; private fixed memory per rank at 10 mm n=4 falls by at least 30 percent, or the floor is attributed with evidence to something XCALibre cannot remove and the milestone says so; per-rank peak at 5 mm n=2 recorded against 1142/1162 MB (D99); serial load time within 10 percent of today's; gate green; every verdict run under five minutes.

## Open questions

- SETTLED by S1 (D102): PSS lowers the n=4 runtime cost by 35 percent, not half.
- Whether a precompile workload that runs a distributed case needs MPI at precompile time; S3 settles it if runtime compilation is the largest private item.
