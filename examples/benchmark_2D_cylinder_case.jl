using XCALibre
using CUDA
using Adapt
using Printf

function build_backend(name::String)
    if name == "cpu"
        return CPU(), 1024
    elseif name == "cuda"
        CUDA.functional() || error("CUDA.functional() is false")
        return CUDABackend(), 32
    else
        error("Unknown backend $name")
    end
end

function default_benchmark_configs()
    return [
        (backend="cpu", mode="baseline", iterations=5, warmup_iterations=1),
        (backend="cpu", mode="amg_example", iterations=5, warmup_iterations=1),
        (backend="cpu", mode="amg_cg_stageb", iterations=5, warmup_iterations=1),
        (backend="cuda", mode="baseline", iterations=1, warmup_iterations=1),
        (backend="cuda", mode="amg_example", iterations=1, warmup_iterations=1),
    ]
end

function amg_pressure_solver(mode::String)
    if mode == "amg" || mode == "amg_example"
        return SolverSetup(
            solver=AMG(
                mode=:solver,
                coarsening=SmoothAggregation(),
                smoother=AMGJacobi(),
                cycle=:V,
                max_levels=8,
                smoothing_steps=10,
                max_coarse_rows=100,
                adaptive_rebuild_factor=1.1
            ),
            preconditioner=Jacobi(),
            convergence=1e-7,
            relax=1.0,
            itmax=40,
            rtol=0.0,
            atol=1e-5
        )
    elseif mode == "amg_stageb"
        return SolverSetup(
            solver=AMG(
                mode=:solver,
                cycle=:W,
                coarsening=SmoothAggregation(),
                smoother=AMGSymmetricGaussSeidel(),
                max_levels=8,
                presweeps=2,
                postsweeps=2,
                max_coarse_rows=100,
                adaptive_rebuild_factor=0.9
            ),
            preconditioner=Jacobi(),
            convergence=1e-7,
            relax=1.0,
            itmax=40,
            rtol=0.0,
            atol=1e-5
        )
    elseif mode == "amg_stageb_jacobi"
        return SolverSetup(
            solver=AMG(
                mode=:solver,
                cycle=:W,
                coarsening=SmoothAggregation(),
                smoother=AMGJacobi(),
                max_levels=8,
                presweeps=2,
                postsweeps=2,
                max_coarse_rows=100,
                adaptive_rebuild_factor=0.9
            ),
            preconditioner=Jacobi(),
            convergence=1e-7,
            relax=1.0,
            itmax=40,
            rtol=0.0,
            atol=1e-5
        )
    elseif mode == "amg_cg_stageb"
        return SolverSetup(
            solver=AMG(
                mode=:cg,
                cycle=:V,
                coarsening=SmoothAggregation(),
                smoother=AMGSymmetricGaussSeidel(),
                max_levels=8,
                presweeps=2,
                postsweeps=2,
                max_coarse_rows=100,
                adaptive_rebuild_factor=0.9
            ),
            preconditioner=Jacobi(),
            convergence=1e-7,
            relax=1.0,
            itmax=40,
            rtol=0.0,
            atol=1e-5
        )
    else
        error("Unknown AMG benchmark mode $mode")
    end
end

function make_case(backend, workgroup, mode::String, iterations)
    grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
    mesh_file = joinpath(grids_dir, "cylinder_d10mm_5mm.unv")
    mesh = UNV2D_mesh(mesh_file, scale=0.001)

    backend isa CPU && activate_multithread(backend)
    hardware = Hardware(backend=backend, workgroup=workgroup)
    mesh_dev = adapt(backend, mesh)

    velocity = [0.5, 0.0, 0.0]
    noSlip = [0.0, 0.0, 0.0]
    nu = 1e-3

    model = Physics(
        time=Transient(),
        fluid=Fluid{Incompressible}(nu=nu),
        turbulence=RANS{Laminar}(),
        energy=Energy{Isothermal}(),
        domain=mesh_dev
    )

    BCs = assign(
        region=mesh_dev,
        (
            U = [
                Dirichlet(:inlet, velocity),
                Zerogradient(:outlet),
                Wall(:cylinder, noSlip),
                Extrapolated(:bottom),
                Extrapolated(:top)
            ],
            p = [
                Zerogradient(:inlet),
                Dirichlet(:outlet, 0.0),
                Wall(:cylinder),
                Extrapolated(:bottom),
                Extrapolated(:top)
            ]
        )
    )

    psolver = if mode == "baseline"
        SolverSetup(
            solver=Cg(),
            preconditioner=Jacobi(),
            convergence=1e-7,
            relax=1.0,
            rtol=0.0,
            atol=1e-5
        )
    elseif startswith(mode, "amg")
        amg_pressure_solver(mode)
    else
        error("Unknown mode $mode")
    end

    solvers = (
        U = SolverSetup(
            solver=Bicgstab(),
            preconditioner=Jacobi(),
            convergence=1e-7,
            relax=1.0,
            rtol=0.0,
            atol=1e-5
        ),
        p = psolver
    )

    schemes = (
        U = Schemes(time=CrankNicolson, divergence=LUST, gradient=Gauss),
        p = Schemes(time=CrankNicolson, gradient=Gauss)
    )

    runtime = Runtime(iterations=iterations, write_interval=-1, time_step=0.0025)
    config = Configuration(solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

    initialise!(model.momentum.U, velocity)
    initialise!(model.momentum.p, 0.0)
    return model, config
end

function run_case(backend_name::String, mode::String, iterations::Integer, warmup_iterations::Integer)
    backend, workgroup = build_backend(backend_name)

    backend isa CUDABackend && (CUDA.reclaim(); CUDA.synchronize())
    GC.gc(true)

    disable_solve_history!()
    reset_solve_history!()
    warm_model, warm_config = make_case(backend, workgroup, mode, warmup_iterations)
    run!(warm_model, warm_config)
    backend isa CUDABackend && (CUDA.synchronize(); CUDA.reclaim())
    GC.gc(true)

    enable_solve_history!()
    reset_solve_history!()
    model, config = make_case(backend, workgroup, mode, iterations)
    residuals = nothing
    elapsed = @elapsed residuals = run!(model, config)
    backend isa CUDABackend && CUDA.synchronize()
    history = solve_history()
    disable_solve_history!()

    result = (
        backend=backend_name,
        mode=mode,
        iterations=length(residuals.p),
        elapsed_s=elapsed,
        p_first=residuals.p[1],
        p_last=residuals.p[end],
        ux_last=residuals.Ux[end],
        pressure_history=[entry for entry in history if entry.equation_kind == "ScalarModel"]
    )

    @printf(
        "RESULT backend=%s mode=%s iterations=%d elapsed_s=%.6f p_first=%.6e p_last=%.6e ux_last=%.6e\n",
        result.backend,
        result.mode,
        result.iterations,
        result.elapsed_s,
        result.p_first,
        result.p_last,
        result.ux_last
    )

    for (solve_id, entry) in enumerate(result.pressure_history)
        @printf(
            "LINEAR backend=%s mode=%s solve=%d solver=%s solver_mode=%s iterations=%d hit_itmax=%s residual0=%.6e residualN=%.6e relN=%.6e\n",
            result.backend,
            result.mode,
            solve_id,
            entry.solver,
            entry.solver_mode,
            entry.iterations,
            string(entry.hit_itmax),
            entry.residual_abs[1],
            entry.final_residual,
            entry.final_relative_residual
        )
    end

    return result
end

function run_benchmarks(configs)
    results = NamedTuple[]
    for cfg in configs
        push!(results, run_case(cfg.backend, cfg.mode, cfg.iterations, cfg.warmup_iterations))
    end
    return results
end

function parse_cli_config(args)
    backend = get(args, 1, "cpu")
    mode = get(args, 2, "baseline")
    iterations = length(args) >= 3 ? parse(Int, args[3]) : 5
    warmup_iterations = length(args) >= 4 ? parse(Int, args[4]) : 1
    return (backend=backend, mode=mode, iterations=iterations, warmup_iterations=warmup_iterations)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_case(parse_cli_config(ARGS)...)
end
