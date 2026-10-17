using XCALibre

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "backwardFacingStep_10mm.unv"

mesh_file = joinpath(grids_dir, grid)
mesh = UNV2D_mesh(mesh_file, scale=0.001)
@test typeof(mesh) <: Mesh2

workgroup = length(mesh.cells) ÷ Threads.nthreads() # workgroupsize(mesh)
backend = CPU()
mesh_dev = adapt(backend, mesh)

Umag = 0.5
velocity = [Umag, 0.0, 0.0]
nu = 1e-3
Re = velocity[1]*0.1/nu

function run_bfs(tscheme)
    model = Physics(
        time = Transient(),
        fluid = Fluid{Incompressible}(nu=nu),
        turbulence = RANS{Laminar}(),
        energy = Energy{Isothermal}(),
        domain = mesh_dev
    )
    BCs = assign(
        region=mesh_dev,
        (
            U = [
                Dirichlet(:inlet, velocity),
                Extrapolated(:outlet),
                Wall(:wall, [0.0, 0.0, 0.0]),
                Dirichlet(:top, [0.0, 0.0, 0.0])
            ],
            p = [
                Extrapolated(:inlet),
                Dirichlet(:outlet, 0.0),
                Extrapolated(:wall),
                Extrapolated(:top)
            ]
        )
    )
    schemes = (U = Schemes(time=tscheme), p = Schemes(time=tscheme))
    solvers = (
        U = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-7, relax=1.0, rtol=1e-3),
        p = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-7, relax=1.0, rtol=1e-3)
    )
    # impulsive start from a non-solenoidal field: Euler and CrankNicolson differ by ~40% at 150
    # steps, the start-up transient being unresolved in Δt; by 400 steps both are within 4%
    runtime = Runtime(iterations=400, time_step=0.005, write_interval=400)
    hardware = Hardware(backend=backend, workgroup=workgroup)
    config = Configuration(
        solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

    GC.gc()

    @test initialise!(model.momentum.U, velocity) === nothing
    @test initialise!(model.momentum.p, 0.0) === nothing

    run!(model, config; progress=false)

    inlet = boundary_average(:inlet, model.momentum.U, BCs.U, config)
    outlet = boundary_average(:outlet, model.momentum.U, BCs.U, config)
    @test outlet ≈ 0.5*inlet atol=0.1*Umag
    return model
end

@testset "Euler" begin
    global modelE = run_bfs(Euler)
end
@testset "CrankNicolson" begin
    modelCN = run_bfs(CrankNicolson)
    @test mean(modelCN.momentum.U.x.values) ≈ mean(modelE.momentum.U.x.values) rtol=0.06
end
