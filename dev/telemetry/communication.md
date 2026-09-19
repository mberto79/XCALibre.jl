# Communication rounds and reductions (P1-M22)

Machine: this laptop, `dev/petscenv_stock`, Julia 1.13, CPU, clock unpinned. Counters: `test/distributed/test_perf.jl` (2D BFS, laminar SIMPLE; 3D budgets are one higher each). Hashes: `dev/scripts/mem_probe.jl worker` on 10 mm offline parts, 2 iterations, Jacobi. Raw logs: `~/.cache/xcal_m22/` (not tracked).

## baseline (HEAD 8b769eb0)

- Per 2D SIMPLE iteration: 7 exchanges, 6 all-reduces (3D: 8, 8).
- `test_perf.jl` laplace halo allocation: 4016 B at n=2, 6544 B at n=3.
- Hashes: 10 mm n=2 `2a522e3ce43de833`, n=4 `dbc3c69ab48b3394` (the latter matches `fixed_rank_memory.md`).

## S1: one schedule per mesh, tags per width

- Counters unchanged (7, 6); halo allocation 2928 B at n=2 (−1088), 5456 B at n=3 (−1088); hashes identical at n=2 and n=4.
- Gate 10/10 in 204 s; `test_halo`, `test_ghosts`, `test_f32`, `test_perf` green at n=2,3.
- `test_gpu.jl` n=1,2 on `dev/petscenv_conda_ompi` green (2/2, 251 s).

## S2: one width-3 exchange for the momentum components

- Exchanges per 2D iteration 7 → 6 (3D 8 → 6), all-reduces unchanged; `check_ghosts` on U after the vector solve zero; hashes identical at n=2 and n=4; gate 10/10 in 200 s; halo, ghosts, f32, perf green at n=2,3.
- `test_perf.jl` vector-solve allocation 47264 → 49104 B at n=2 (two extra kernel launches); budget 98304.
- n=8 hash not run: language server resident (2.8 GB) and no 10 mm n=8 parts; n=4 stands in, as R8 makes them one bar.
- `test_gpu.jl` n=1,2 green (2/2, 254 s).

## S3: one all-reduce per equation for residuals

- All-reduces per iteration 6 → 2 (3D 8 → 2); exchanges 6; hashes identical at n=2 and n=4; laplace `residual` allocation 368 → 304 B; gate 10/10 in 200 s; halo, ghosts, f32, perf green at n=2,3; `test_gpu.jl` n=1,2 green (254 s).

## S4: one width-4 exchange for rD and Hv (SIMPLE body)

- Exchanges per iteration 6 → 5, all-reduces 2; hashes identical at n=2 and n=4 on 10 mm parts regenerated at part format 2 (format 1 parts are refused at the header); gate 10/10 in 200 s; offline, io, halo, f32, perf green at n=2,3.
- PISO keeps a separate exchange for rD and one for Hv per corrector: `H!` runs once per corrector, so rD cannot share the first one without moving `interpolate!(rDf, rD)` into the corrector loop.
- `check_ghosts` zero on rD and Hv after the paired exchange (`test_ghosts.jl` now mirrors the body) at n=2,3; `test_gpu.jl` n=1,2 green (255 s).

## S5: persistent requests

- Halo allocation 2928 → 2736 B at n=2 (5456 → 5200 at n=3); what remains is the pack/unpack launches, so the `test_perf.jl` budget drops to 512 + 3072 per neighbour.
- `dev/scripts/halo_bench.jl` at 10 mm n=8, 200 reps × 10 alternated rounds, medians: width 3 persistent 16.57 µs vs fresh 16.89 µs; width 1 11.89 vs 12.05 µs.
- Hashes identical at n=2 and n=4; gate 10/10 in 197 s; halo, ghosts, f32, perf green at n=2,3; `test_gpu.jl` n=1,2 green (256 s, covers the host-staged path).
