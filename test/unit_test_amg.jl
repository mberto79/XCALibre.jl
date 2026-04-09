using XCALibre
using LinearAlgebra
using SparseMatricesCSR
using KernelAbstractions

println("=== AMG Unit Test ===")

# ── Build a small 2D Poisson matrix ─────────────────────────────────────────
# 5×5 grid interior → 25 DOF (Laplacian on uniform grid)

function build_poisson(n)
    # n × n interior Laplacian stencil (5-point)
    N = n * n
    rows = Int[]; cols = Int[]; vals = Float64[]
    for i in 1:n, j in 1:n
        id = (i-1)*n + j
        push!(rows, id); push!(cols, id); push!(vals, 4.0)
        if i > 1; r = (i-2)*n+j; push!(rows,id); push!(cols,r); push!(vals,-1.0); end
        if i < n; r = (i  )*n+j; push!(rows,id); push!(cols,r); push!(vals,-1.0); end
        if j > 1; push!(rows,id); push!(cols,id-1); push!(vals,-1.0); end
        if j < n; push!(rows,id); push!(cols,id+1); push!(vals,-1.0); end
    end
    A = sparsecsr(rows, cols, vals, N, N)
    return XCALibre.Multithread.SparseXCSR(A)
end

n  = 10
A  = build_poisson(n)
N  = n * n
b  = randn(Float64, N)

# ── Build AMGWorkspace directly (bypass ModelEquation) ────────────────────────
using XCALibre.Solve

backend  = CPU()
workgroup = 64

amg_opts = AMG(
    smoother     = JacobiSmoother(2, 2/3, zeros(N)),
    cycle        = VCycle(),
    max_levels   = 10,
    coarsest_size = 8,
    pre_sweeps   = 2,
    post_sweeps  = 2,
    strength     = 0.25,
    coarsening   = :SA,
)

ws = _workspace(amg_opts, b)

# Force setup
XCALibre.Solve.amg_setup!(ws, A, backend, workgroup)

println("Number of levels built: ", length(ws.levels))
println("Coarsest level size:    ", length(ws.levels[end].b))

# ── Solve A x = b ─────────────────────────────────────────────────────────────
ws.levels[1].b .= b
ws.levels[1].x .= 0.0

r0_norm = norm(b)

for iter in 1:50
    XCALibre.Solve.run_cycle!(ws.levels, ws.opts, ws.opts.cycle, backend, workgroup)
    XCALibre.Solve.amg_residual!(ws.levels[1].r, ws.levels[1].A,
                                  ws.levels[1].x, ws.levels[1].b,
                                  backend, workgroup)
    res = norm(ws.levels[1].r) / r0_norm
    if iter == 1 || iter % 5 == 0 || res < 1e-8
        println("  iter $iter: relative residual = $res")
    end
    res < 1e-8 && break
end

final_res = norm(b - A * ws.levels[1].x) / norm(b)
println("Final relative residual: $final_res")
@assert final_res < 1e-6 "AMG did not converge to tolerance 1e-6 (got $final_res)"

println("\n✓  AMG unit test PASSED")

# ── Test update! (simulate outer-iteration coefficient change) ─────────────────
println("\n--- Testing update! ---")
ws2 = _workspace(amg_opts, b)
XCALibre.Solve.amg_setup!(ws2, A, backend, workgroup)
n_levels_before = length(ws2.levels)

# Simulate a small coefficient perturbation (same sparsity)
A_new = build_poisson(n)

XCALibre.Solve.update!(ws2, A_new, backend, workgroup)
n_levels_after = length(ws2.levels)

@assert n_levels_before == n_levels_after "update! must not change the number of levels"
println("Levels before/after update: $n_levels_before / $n_levels_after  ✓")

# Solve again after update
ws2.levels[1].b .= b
ws2.levels[1].x .= 0.0
for _ in 1:50
    XCALibre.Solve.run_cycle!(ws2.levels, ws2.opts, ws2.opts.cycle, backend, workgroup)
    XCALibre.Solve.amg_residual!(ws2.levels[1].r, ws2.levels[1].A,
                                   ws2.levels[1].x, ws2.levels[1].b,
                                   backend, workgroup)
    norm(ws2.levels[1].r) / r0_norm < 1e-8 && break
end
final_res2 = norm(b - A_new * ws2.levels[1].x) / norm(b)
@assert final_res2 < 1e-6 "AMG after update! did not converge (got $final_res2)"
println("Post-update residual: $final_res2  ✓")

println("\n=== All AMG tests PASSED ===")
