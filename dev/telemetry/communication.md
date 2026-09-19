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
