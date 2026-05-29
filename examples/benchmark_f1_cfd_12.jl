using XCALibre
using Adapt
using CUDA
using JLD2
using Printf

function f1_backend(name::AbstractString)
    if lowercase(name) == "cuda"
        CUDA.functional() || error("CUDA.functional() is false")
        return CUDABackend(), 32
    end
    backend = CPU()
    activate_multithread(backend)
    return backend, 1024
end

function f1_pressure_solver(mode::AbstractString, ftype)
    if mode == "baseline"
        return SolverSetup(
            float_type=ftype,
            solver=Cg(),
            preconditioner=Jacobi(),
            convergence=1e-5,
            relax=0.3,
            rtol=1e-2,
            atol=1e-15
        )
    elseif mode == "amg"
        return SolverSetup(
            float_type=ftype,
            solver=AMG(),
            preconditioner=Jacobi(),
            convergence=1e-5,
            relax=0.3,
            rtol=1e-2,
            atol=1e-15
        )
    end
    error("Unknown F1 benchmark mode $mode")
end

function make_f1_case(case_dir::AbstractString, backend, workgroup, mode::AbstractString, iterations::Integer)
    mesh = load_object(joinpath(case_dir, "CRMHL.jld2"))
    ftype = Float64

    hardware = Hardware(backend=backend, workgroup=workgroup)
    mesh_dev = adapt(backend, mesh)

    U = 35
    velocity = [0.0, 0.0, -U]
    nu = 1.5e-5
    k = 1.5 * (U * 0.01)^2
    omega = k / (50 * nu)
    nut = k / omega

    model = Physics(
        time=Steady(),
        fluid=Fluid{Incompressible}(nu=nu),
        turbulence=RANS{KOmega}(),
        energy=Energy{Isothermal}(),
        domain=mesh_dev
    )

    symmetry_patches = [:SYM, :TOP, :BOTTOM]
    symmetry = Symmetry.(symmetry_patches)
    BCs = assign(
        region=mesh_dev,
        (
            U=[
                Dirichlet(:IN, velocity),
                Zerogradient(:OUT),
                Wall(:F1, [0.0, 0.0, 0.0]),
                symmetry...
            ],
            p=[
                Zerogradient(:IN),
                Dirichlet(:OUT, 0.0),
                Wall(:F1),
                symmetry...
            ],
            k=[
                Dirichlet(:IN, k),
                Zerogradient(:OUT),
                KWallFunction(:F1),
                symmetry...
            ],
            omega=[
                Dirichlet(:IN, omega),
                Zerogradient(:OUT),
                OmegaWallFunction(:F1),
                symmetry...
            ],
            nut=[
                Extrapolated(:IN),
                Extrapolated(:OUT),
                NutWallFunction(:F1),
                symmetry...
            ]
        )
    )

    schemes = (
        U=Schemes(divergence=Upwind, gradient=Gauss, limiter=nothing),
        k=Schemes(divergence=Upwind, gradient=Gauss, limiter=nothing),
        omega=Schemes(divergence=Upwind, gradient=Gauss, limiter=nothing),
        p=Schemes(gradient=Gauss, limiter=nothing)
    )

    solvers = (
        U=SolverSetup(
            float_type=ftype,
            solver=Bicgstab(),
            preconditioner=Jacobi(),
            convergence=1e-5,
            relax=0.7,
            rtol=1e-1,
            atol=1e-15
        ),
        p=f1_pressure_solver(mode, ftype),
        k=SolverSetup(
            float_type=ftype,
            solver=Bicgstab(),
            preconditioner=Jacobi(),
            convergence=1e-5,
            relax=0.6,
            rtol=1e-1,
            atol=1e-15
        ),
        omega=SolverSetup(
            float_type=ftype,
            solver=Bicgstab(),
            preconditioner=Jacobi(),
            convergence=1e-5,
            relax=0.6,
            rtol=1e-1,
            atol=1e-15
        )
    )

    runtime = Runtime(iterations=iterations, time_step=1, write_interval=-1)
    config = Configuration(solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

    initialise!(model.momentum.U, velocity)
    initialise!(model.momentum.p, 0.0)
    initialise!(model.turbulence.k, k)
    initialise!(model.turbulence.omega, omega)
    initialise!(model.turbulence.nut, nut)
    return model, config
end

function last_velocity_residual(residuals)
    return maximum(abs, (residuals.Ux[end], residuals.Uy[end], residuals.Uz[end]))
end

function run_f1_case(case_dir::AbstractString, backend_name::AbstractString, mode::AbstractString, iterations::Integer)
    backend, workgroup = f1_backend(backend_name)
    backend isa CUDABackend && (CUDA.reclaim(); CUDA.synchronize())
    GC.gc(true)

    disable_solve_history!()
    reset_solve_history!()
    warm_model, warm_config = make_f1_case(case_dir, backend, workgroup, mode, 1)
    run!(warm_model, warm_config)
    backend isa CUDABackend && (CUDA.synchronize(); CUDA.reclaim())
    GC.gc(true)

    enable_solve_history!()
    reset_solve_history!()
    model, config = make_f1_case(case_dir, backend, workgroup, mode, iterations)
    residuals = nothing
    elapsed_s = @elapsed begin
        residuals = run!(model, config)
        backend isa CUDABackend && CUDA.synchronize()
    end
    history = solve_history()
    disable_solve_history!()
    pressure_history = [entry for entry in history if entry.equation_kind == "ScalarModel"]
    pressure_solves = length(pressure_history)
    pressure_linear_iterations = sum(entry.iterations for entry in pressure_history)

    @printf(
        "F1_CFD phase=warmed backend=%s mode=%s iterations=%d elapsed_s=%.6f p_last=%.6e u_last=%.6e ux_last=%.6e uy_last=%.6e uz_last=%.6e pressure_solves=%d pressure_linear_iterations=%d\n",
        backend_name,
        mode,
        length(residuals.p),
        elapsed_s,
        residuals.p[end],
        last_velocity_residual(residuals),
        residuals.Ux[end],
        residuals.Uy[end],
        residuals.Uz[end],
        pressure_solves,
        pressure_linear_iterations
    )
    return residuals
end

function main(args)
    case_dir = get(args, 1, "/home/humberto/casesXCALibre/F1-fetchCFD_Minimal")
    backend_name = get(args, 2, "cuda")
    mode = get(args, 3, "baseline")
    iterations = length(args) >= 4 ? parse(Int, args[4]) : 12
    run_f1_case(case_dir, backend_name, mode, iterations)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
