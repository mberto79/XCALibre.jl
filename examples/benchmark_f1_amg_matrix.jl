using XCALibre
using Adapt
using CUDA
using LinearAlgebra
using SparseArrays
using Krylov
using Printf
using Serialization

function load_pressure_systems(path::AbstractString)
    open(path, "r") do io
        return deserialize(io)
    end
end

function maybe_cuda_backend(name::AbstractString)
    if lowercase(name) == "cuda"
        return CUDABackend(), 32
    end
    backend = CPU()
    activate_multithread(backend)
    return backend, 1024
end

function level_rows(hierarchy)
    return join((string(size(level.A, 1)) for level in hierarchy.levels), "->")
end

function scaled_matrix(A, scale)
    B = deepcopy(A)
    XCALibre.Solve._nzval(B) .*= scale
    return B
end

function sync_backend(backend)
    if string(nameof(typeof(backend))) == "CUDABackend"
        CUDA.synchronize()
    end
    return nothing
end

function bench_system(system; backend, workgroup, solver=AMG(), itmax=200, atol=1e-15, rtol=1e-2, refresh_updates=3)
    hardware = Hardware(backend=backend, workgroup=workgroup)
    config = (hardware=hardware,)
    b = adapt(backend, copy(system.b))
    x = adapt(backend, copy(system.x0))
    workspace = XCALibre.Solve._workspace(solver, b)

    build_s = @elapsed begin
        XCALibre.Solve.update!(workspace, system.A, solver, config)
        sync_backend(backend)
    end

    hierarchy = workspace.hierarchy
    fine_A = hierarchy.levels[1].A
    solve_s = @elapsed begin
        XCALibre.Solve.amg_cg_solve!(workspace, hierarchy, solver, fine_A, b, x; itmax, atol, rtol)
        sync_backend(backend)
    end

    refresh_s = 0.0
    for k in 1:refresh_updates
        A_update = scaled_matrix(system.A, 1 + 0.001 * k)
        refresh_s += @elapsed begin
            XCALibre.Solve.update!(workspace, A_update, solver, config)
            sync_backend(backend)
        end
    end

    timing = workspace.timing
    return (
        rows=level_rows(hierarchy),
        operator_complexity=hierarchy.operator_complexity,
        grid_complexity=hierarchy.grid_complexity,
        build_s=build_s,
        solve_s=solve_s,
        refresh_s=refresh_s,
        build_calls=timing.build_calls,
        refresh_calls=timing.refresh_calls,
        finest_refresh_calls=timing.finest_refresh_calls,
        apply_s=timing.apply_time_s,
        apply_calls=timing.apply_calls,
        coarse_rhs_copy_s=hierarchy.coarse_rhs_copy_time_s,
        coarse_cpu_solve_s=hierarchy.coarse_cpu_solve_time_s,
        coarse_x_copy_s=hierarchy.coarse_x_copy_time_s,
        coarse_solve_calls=hierarchy.coarse_solve_calls,
        coarse_device_solve_s=hierarchy.coarse_device_solve_time_s,
        coarse_device_solve_calls=hierarchy.coarse_device_solve_calls,
        iterations=workspace.iterations,
        final_relative=workspace.last_relative_residual
    )
end

function bench_baseline(system; backend, itmax=1000, atol=1e-15, rtol=1e-2)
    string(nameof(typeof(backend))) == "CUDABackend" || return nothing
    i, j, v = XCALibre.Solve._csr_triplets(system.A)
    n = size(system.A, 1)
    Acsc = sparse(i, j, v, n, n)
    A = CUDA.CUSPARSE.CuSparseMatrixCSR(Acsc)
    b = adapt(backend, copy(system.b))
    dvec = collect(diag(Acsc))
    invdiag = adapt(backend, eltype(dvec)[d != 0 ? inv(d) : one(eltype(dvec)) for d in dvec])
    M = Diagonal(invdiag)
    Krylov.cg(A, b; M=M, ldiv=false, itmax=itmax, atol=atol, rtol=rtol)  # warm
    CUDA.synchronize()
    local stats
    solve_s = @elapsed begin
        _, stats = Krylov.cg(A, b; M=M, ldiv=false, itmax=itmax, atol=atol, rtol=rtol)
        CUDA.synchronize()
    end
    return (solve_s=solve_s, iterations=stats.niter)
end

function print_result(system_id, phase, result)
    @printf(
        "F1_AMG_MATRIX phase=%s system=%d rows=%s iterations=%d final_relative=%.6e build_s=%.6e solve_s=%.6e refresh_s=%.6e apply_s=%.6e apply_calls=%d coarse_rhs_copy_s=%.6e coarse_cpu_solve_s=%.6e coarse_x_copy_s=%.6e coarse_solve_calls=%d coarse_device_solve_s=%.6e coarse_device_solve_calls=%d refresh_calls=%d finest_refresh_calls=%d operator_complexity=%.6f grid_complexity=%.6f\n",
        phase,
        system_id,
        result.rows,
        result.iterations,
        result.final_relative,
        result.build_s,
        result.solve_s,
        result.refresh_s,
        result.apply_s,
        result.apply_calls,
        result.coarse_rhs_copy_s,
        result.coarse_cpu_solve_s,
        result.coarse_x_copy_s,
        result.coarse_solve_calls,
        result.coarse_device_solve_s,
        result.coarse_device_solve_calls,
        result.refresh_calls,
        result.finest_refresh_calls,
        result.operator_complexity,
        result.grid_complexity
    )
end

function main(args)
    path = get(args, 1, "/home/humberto/casesXCALibre/F1-fetchCFD_Minimal/f1_pressure_systems.bin")
    backend_name = get(args, 2, "cuda")
    samples = length(args) >= 3 ? parse(Int, args[3]) : 1
    max_coarse_rows = length(args) >= 4 ? parse(Int, args[4]) : 4096
    coarse_refresh_interval = length(args) >= 5 ? parse(Int, args[5]) : 20
    warmed_runs = length(args) >= 6 ? parse(Int, args[6]) : 1
    solver = AMG(max_coarse_rows=max_coarse_rows, coarse_refresh_interval=coarse_refresh_interval)
    backend, workgroup = maybe_cuda_backend(backend_name)
    systems = load_pressure_systems(path)
    for (system_id, system) in enumerate(Iterators.take(systems, samples))
        cold_result = bench_system(system; backend, workgroup, solver)
        print_result(system_id, "cold", cold_result)
        local last_warmed = cold_result
        for run in 1:warmed_runs
            phase = warmed_runs == 1 ? "warmed" : "warmed_$run"
            last_warmed = bench_system(system; backend, workgroup, solver)
            print_result(system_id, phase, last_warmed)
        end
        baseline = bench_baseline(system; backend)
        if baseline !== nothing
            @printf(
                "F1_AMG_BASELINE system=%d cg_jacobi_solve_s=%.6e cg_jacobi_iterations=%d amg_solve_s=%.6e amg_iterations=%d speedup=%.3f\n",
                system_id, baseline.solve_s, baseline.iterations,
                last_warmed.solve_s, last_warmed.iterations,
                baseline.solve_s / last_warmed.solve_s
            )
        end
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
