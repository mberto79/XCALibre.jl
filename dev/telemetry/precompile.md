# Shipped precompilation (P1-M26)

Machine: this laptop, `dev/petscenv_stock` (CPU) and `dev/petscenv_conda_ompi` (GPU, RTX 4070 Laptop), Julia 1.13. Raw logs: `~/.cache/xcal_m26/` (not tracked).

## S1 size-free kernel launches

- 166 kernel constructions moved from `f!(_setup(backend, workgroup, n)...)` to `_sized(f!, backend, workgroup, n)`: an integer workgroup stays static, the range is passed at launch; under `AutoTune` both are passed at launch.
- Second mesh size in one session (`dev/scripts/compile_sizes.jl`, 3D box 5 then 10, laminar SIMPLE 3 iterations): 3.28 s before, 0.01 s after; first run 8.07 → 7.49-7.82 s.
- Residual hashes at 10 mm, 20 iterations, Jacobi: n=2 `66cf6fd5c0c335be`, n=4 `550bb695b7dbab9c`, both unchanged; 200 iterations n=2 `a97e2bb832756101` unchanged.
- CPU time, 10 mm n=2, 200 iterations, warm `run!`: 6.00/6.10/6.11 s before, 6.01/6.09/6.06 s after (static workgroup kept for integer workgroups; with a dynamic workgroup it was 6.10/6.22/6.18).
- GPU time, 10 mm n=1, `scaling_probe.jl dev=cuda`: 8.2 ms per iteration before, 8.0 and 8.0 after; residuals agree to about 1e-14 (GPU is tolerance-checked).
- Private memory after `run!`, rank 0/1, forced GC (`mem_probe.jl gc=1`): `AutoTune` 600/594 → 637/624 MB (+33, first-run compilation garbage, live heap equal); workgroup 64: 591/601 → 605/606 MB (noise).
- gate 10/10; serial SIMPLE, PISO and k-omega cases 20/20; `test_gpu.jl` n=1 green.
