# Phase 0 baseline harness for the AMG fused-pipeline project.
# Times the CURRENT AMG(mode=AMGSolver(), coarsening=Geometric()) on synthetic 3D Poisson.
# This is the regression oracle for all later phases. Run via the persistent Julia REPL.
#
# Usage (in REPL, project active):
#   include("src/Solve/AMG/bench/amg_bench.jl")
#   amg_bench_run(CPU())                       # CPU baseline
#   using CUDA; amg_bench_run(CUDABackend())   # GPU baseline (needs CUDA ext loaded)
#
# Sizes capped for an 8 GB GPU / ~7.6 GB host free: n=32,48,64 (n^3 unknowns).

using XCALibre
using SparseArrays, LinearAlgebra
using XCALibre.Solve: _workspace, update!, _amg_solve_mode!, _rowptr, _colval, _nzval
using XCALibre: SparseXCSR, sparsecsr
using KernelAbstractions
import Adapt: adapt

# Move the host CSR onto `backend` WITHOUT densifying. Generic `adapt(CUDABackend(), sparse)`
# converts to a dense CuArray (OOM); the real solver keeps it as a device CuSparseMatrixCSR.
# References CUDA at call-time only (no compile dep) — call the GPU branch only with CUDA loaded.
_bench_to_device(::CPU, A_host) = A_host
function _bench_to_device(backend, A_host)
    rp = Int32.(collect(_rowptr(A_host))); cv = Int32.(collect(_colval(A_host)))
    nz = collect(_nzval(A_host)); m, n = size(A_host)
    return CUDA.CUSPARSE.CuSparseMatrixCSR(CUDA.CuVector(rp), CUDA.CuVector(cv), CUDA.CuVector(nz), (m, n))
end

# 7-point Laplacian on an n x n x n grid (Dirichlet), n^3 unknowns. Returns (SparseXCSR, rhs).
function poisson3d(n=32)
    N = n^3
    lin(i, j, k) = ((k - 1) * n + (j - 1)) * n + i
    rows = Int[]; cols = Int[]; vals = Float64[]
    sizehint!(rows, 7N); sizehint!(cols, 7N); sizehint!(vals, 7N)
    @inbounds for k in 1:n, j in 1:n, i in 1:n
        d = lin(i, j, k)
        push!(rows, d); push!(cols, d); push!(vals, 6.0)
        i > 1 && (push!(rows, d); push!(cols, lin(i-1, j, k)); push!(vals, -1.0))
        i < n && (push!(rows, d); push!(cols, lin(i+1, j, k)); push!(vals, -1.0))
        j > 1 && (push!(rows, d); push!(cols, lin(i, j-1, k)); push!(vals, -1.0))
        j < n && (push!(rows, d); push!(cols, lin(i, j+1, k)); push!(vals, -1.0))
        k > 1 && (push!(rows, d); push!(cols, lin(i, j, k-1)); push!(vals, -1.0))
        k < n && (push!(rows, d); push!(cols, lin(i, j, k+1)); push!(vals, -1.0))
    end
    A = SparseXCSR(sparsecsr(rows, cols, vals, N, N))
    return A, ones(Float64, N)
end

function _bench_config(backend, workgroup)
    setup = SolverSetup(solver=AMG(mode=AMGSolver(), coarsening=Geometric()),
                        preconditioner=Jacobi(), convergence=1e-7, relax=1.0, rtol=1e-8, atol=1e-8)
    Configuration(solvers=(T=setup,), schemes=(T=Schemes(),),
                  runtime=Runtime(iterations=1, write_interval=-1, time_step=1.0),
                  hardware=Hardware(backend=backend, workgroup=workgroup),
                  boundaries=(T=(),))
end

# Median wall time over `reps` (compile excluded by a warmup call).
function _timed(f, reps)
    f()  # warmup / compile
    ts = Float64[]
    for _ in 1:reps
        t0 = time_ns(); f(); push!(ts, (time_ns() - t0) / 1e9)
    end
    return sort(ts)[cld(length(ts), 2)]
end

# One baseline measurement: setup time, total solve time, iters, final rel residual.
function amg_bench_one(backend, n; workgroup=64, itmax=500, reps=3)
    A_h, b_h = poisson3d(n)
    solver = AMG(mode=AMGSolver(), coarsening=Geometric())
    config = _bench_config(backend, workgroup)
    A = _bench_to_device(backend, A_h)
    b = backend isa CPU ? b_h : adapt(backend, b_h)

    setup_t = _timed(reps) do
        ws = _workspace(solver, b)
        update!(ws, A, solver, config)
        backend isa CPU || KernelAbstractions.synchronize(backend)
    end

    ws = _workspace(solver, b)
    update!(ws, A, solver, config)
    solve_t = _timed(reps) do
        x = similar(b); fill!(x, 0)
        _amg_solve_mode!(ws, ws.hierarchy, solver, solver.mode,
                         ws.hierarchy.levels[1].A, b, x; itmax=itmax, atol=1e-8, rtol=1e-8)
        backend isa CPU || KernelAbstractions.synchronize(backend)
    end

    x = similar(b); fill!(x, 0)
    _amg_solve_mode!(ws, ws.hierarchy, solver, solver.mode,
                     ws.hierarchy.levels[1].A, b, x; itmax=itmax, atol=1e-8, rtol=1e-8)
    iters = ws.iterations
    relres = ws.last_relative_residual
    percycle = iters > 0 ? solve_t / iters : solve_t
    return (; n, N=n^3, setup_s=setup_t, solve_s=solve_t, percycle_s=percycle, iters, relres)
end

function amg_bench_run(backend; ns=(32, 48, 64), kwargs...)
    println("AMG baseline | backend=$(typeof(backend)) | AMGSolver + Geometric")
    println("n     N         setup_s    solve_s    percycle_s  iters  relres")
    rows = NamedTuple[]
    for n in ns
        r = amg_bench_one(backend, n; kwargs...)
        push!(rows, r)
        println(rpad(r.n, 5), " ", rpad(r.N, 9), " ",
                rpad(round(r.setup_s, sigdigits=4), 10), " ",
                rpad(round(r.solve_s, sigdigits=4), 10), " ",
                rpad(round(r.percycle_s, sigdigits=4), 11), " ",
                rpad(r.iters, 6), " ", round(r.relres, sigdigits=4))
    end
    return rows
end
