using XCALibre
using LinearAlgebra
# using CUDA

# Mixed-precision linear solves on a periodic 3D cascade: the same steady run with the linear systems
# solved in Float64 and with their corrections solved in Float32 (MixedF32) or Float16 (MixedF16).
# Fields, residuals and the outer SIMPLE iteration stay in Float64 throughout.

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh_file = joinpath(grids_dir, "cascade_3D_periodic_4mm.unv")
mesh = UNV3D_mesh(mesh_file, scale=0.001)

# backend = CUDABackend(); workgroup = 32
backend = CPU(); workgroup = 1024; activate_multithread(backend)

hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

periodic = construct_periodic(mesh, backend, :top, :bottom)
sides = Symmetry.([:side1, :side2])

velocity = [0.25, 0.0, 0.0]
nu = 1e-3

function run_case(precision; iterations=300)
    model = Physics(
        time = Steady(),
        fluid = Fluid{Incompressible}(nu=nu),
        turbulence = RANS{Laminar}(),
        energy = Energy{Isothermal}(),
        domain = mesh_dev
        )

    BCs = assign(
        region = mesh_dev,
        (
            U = [
                Dirichlet(:inlet, velocity),
                Zerogradient(:outlet),
                Wall(:plate, [0.0, 0.0, 0.0]),
                periodic...,
                sides...
            ],
            p = [
                Zerogradient(:inlet),
                Dirichlet(:outlet, 0.0),
                Wall(:plate),
                periodic...,
                sides...
            ]
        )
    )

    schemes = (
        U = Schemes(divergence=Linear, gradient=Gauss),
        p = Schemes(gradient=Gauss)
    )

    # MixedF16 supports the Jacobi preconditioner only; MixedF32 accepts any serial preconditioner
    solvers = (
        U = SolverSetup(
            solver = Bicgstab(),
            preconditioner = Jacobi(),
            convergence = 1e-7,
            relax = 0.7,
            rtol = 1e-3,
            precision = precision
        ),
        p = SolverSetup(
            solver = Cg(),
            preconditioner = Jacobi(),
            convergence = 1e-7,
            relax = 0.3,
            rtol = 1e-3,
            precision = precision
        )
    )

    runtime = Runtime(iterations=iterations, time_step=1, write_interval=-1)
    config = Configuration(
        solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

    initialise!(model.momentum.U, velocity)
    initialise!(model.momentum.p, 0.0)

    run!(model, config) # compilation run
    initialise!(model.momentum.U, velocity)
    initialise!(model.momentum.p, 0.0)
    time = @elapsed residuals = run!(model, config)
    (; time, residuals, p=Array(model.momentum.p.values))
end

full = run_case(FullPrecision())
mixed32 = run_case(MixedF32())
# mixed16 = run_case(MixedF16()) # benefits GPUs; on CPUs Float16 arithmetic is slower than Float64

for (name, r) ∈ (("FullPrecision", full), ("MixedF32", mixed32))
    dp = norm(r.p - full.p)/norm(full.p)
    println("$name: $(round(r.time, digits=2)) s, final p residual $(r.residuals.p[end]), p difference $dp")
end
