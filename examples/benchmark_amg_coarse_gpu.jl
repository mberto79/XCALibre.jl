# GPU benchmark for AMG coarse-solve strategies (OnDevice / OnDeviceJacobi / OnDeviceChebyshev /
# OnDeviceKrylov) on captured pressure matrices. Times (coarse REFRESH + solve) per rep — the refresh
# MUST be included because a host-LU refactor on a large coarsest is the cost the on-device solvers
# avoid. Reports outer iters + ms per config. CPU baseline of plain Cg+Jacobi is cross-referenced
# from AMG_OnDeviceKrylov_findings.md (390 ms F1 / 36 ms cyl). Run:
#   julia --project examples/benchmark_amg_coarse_gpu.jl <bin_path> <reps>
# bin_path defaults to the F1 1.68M systems; reps defaults to 10.

using XCALibre
using CUDA
using Adapt
using LinearAlgebra
using Printf
using Serialization

const SolveMod = XCALibre.Solve

load_systems(path) = open(deserialize, path, "r")

# Adapt one captured system (.A SparseXCSR, .b, .x0) to the device backend.
function to_device(system, backend)
    A = adapt(backend, system.A)
    b = adapt(backend, collect(system.b))
    x0 = adapt(backend, collect(system.x0))
    return (A=A, b=b, x0=x0)
end

# Cg-mode AMG configs to compare (coarse_solve strategy is the variable under test).
function cg_configs(max_coarse_rows)
    base(cs) = AMG(mode=Cg(), coarsening=SmoothAggregation(), smoother=AMGJacobi(),
                   max_coarse_rows=max_coarse_rows, coarse_solve=cs)
    return [
        ("OnDevice",            base(OnDevice(max_rows=100000))),
        ("OnDeviceJacobi(10)",  base(OnDeviceJacobi(iterations=10))),
        ("OnDeviceJacobi(20)",  base(OnDeviceJacobi(iterations=20))),
        ("OnDeviceChebyshev(10)", base(OnDeviceChebyshev(degree=10))),
        ("OnDeviceChebyshev(20)", base(OnDeviceChebyshev(degree=20))),
    ]
end

# Solve from x0=0: the captured x0 is a previous-step solution (norm O(100)) inconsistent with the
# tiny correction RHS b, which makes relres=rnorm/bnorm blow up. From 0, relres starts at 1.
function bench_one(sys, solver, backend, workgroup; reps=10, itmax=200, atol=0.0, rtol=1e-2)
    b = sys.b
    workspace = SolveMod._workspace(solver, b)
    hierarchy = SolveMod.setup_hierarchy(sys.A, solver, backend, workgroup; log_diagnostics=false)
    workspace.hierarchy = hierarchy
    z = CUDA.zeros(eltype(b), length(b))
    SolveMod.amg_cg_solve!(workspace, hierarchy, solver, sys.A, b, copy(z); itmax=itmax, atol=atol, rtol=rtol)
    CUDA.synchronize()
    iters = 0
    t = @elapsed begin
        for _ in 1:reps
            SolveMod._refresh_coarse_operators!(hierarchy, solver)  # coarse refresh included
            SolveMod.amg_cg_solve!(workspace, hierarchy, solver, sys.A, b, copy(z); itmax=itmax, atol=atol, rtol=rtol)
            iters = workspace.iterations
        end
        CUDA.synchronize()
    end
    rows = join(map(l -> string(size(l.A, 1)), hierarchy.levels), "->")
    return (ms=1000 * t / reps, iters=iters, relres=workspace.last_relative_residual, rows=rows)
end

function main(args)
    path = get(args, 1, "/home/humberto/casesXCALibre/F1-fetchCFD_Minimal/f1_pressure_systems.bin")
    reps = length(args) >= 2 ? parse(Int, args[2]) : 10
    backend = CUDABackend()
    workgroup = 256

    systems = load_systems(path)
    sys_host = systems[end]
    n = size(sys_host.A, 1)
    @printf("LOADED path=%s systems=%d n=%d\n", path, length(systems), n)
    sys = to_device(sys_host, backend)

    # Truncate high: a few candidate coarsest sizes (set above the second-coarsest size to win).
    for mcr in (8000, 16000, 4096)
        @printf("\n== max_coarse_rows=%d ==\n", mcr)
        for (name, solver) in cg_configs(mcr)
            local r
            try
                r = bench_one(sys, solver, backend, workgroup; reps=reps)
                @printf("RESULT mcr=%d cfg=%-22s ms=%8.2f iters=%4d relres=%.2e rows=%s\n",
                        mcr, name, r.ms, r.iters, r.relres, r.rows)
            catch err
                println("ERROR  mcr=$mcr cfg=$name :: ", sprint(showerror, err))
            end
            flush(stdout)
        end
    end
    return nothing
end

main(ARGS)
