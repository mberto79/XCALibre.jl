using XCALibre

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "flatplate_2D_laminar.unv"), scale=0.001)
@test typeof(mesh) <: Mesh2

workgroup = workgroupsize(mesh)
backend = CPU()
mesh_dev = adapt(backend, mesh)

Umag = 0.2
velocity = [Umag, 0.0, 0.0]
nu = 1e-4
cp = 1005.0
gamma = 1.4
Pr = 0.7
Tin = 300.0
Twall = 310.0

function run_variant(visc; iterations=50)
    model = Physics(
        time = Steady(),
        fluid = Fluid{WeaklyCompressible}(nu=visc, cp=cp, gamma=gamma, Pr=Pr),
        turbulence = RANS{Laminar}(),
        energy = Energy{SensibleEnthalpy}(Tref=288.15),
        domain = mesh_dev
    )
    BCs = assign(
        region=mesh,
        (
            U = [
                Dirichlet(:inlet, velocity),
                Extrapolated(:outlet),
                Wall(:wall, [0.0, 0.0, 0.0]),
                Symmetry(:top)
            ],
            p = [
                Extrapolated(:inlet),
                Dirichlet(:outlet, 100000.0),
                Wall(:wall),
                Symmetry(:top)
            ],
            he = [
                FixedTemperature(:inlet, T=Tin, Enthalpy(cp=cp, Tref=288.15)),
                Extrapolated(:outlet),
                FixedTemperature(:wall, T=Twall, Enthalpy(cp=cp, Tref=288.15)),
                Symmetry(:top)
            ],
        )
    )
    schemes = (
        U = Schemes(divergence=Linear),
        p = Schemes(divergence=Linear),
        he = Schemes(divergence=Linear)
    )
    solvers = (
        U = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-7, relax=0.7, rtol=1e-1),
        p = SolverSetup(solver=Cg(), preconditioner=Jacobi(), convergence=1e-7, relax=0.3, rtol=1e-2),
        he = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-7, relax=0.7, rtol=1e-1)
    )
    runtime = Runtime(iterations=iterations, write_interval=iterations, time_step=1)
    hardware = Hardware(backend=backend, workgroup=workgroup)
    config = Configuration(
        solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

    GC.gc()

    @test initialise!(model.momentum.U, velocity) === nothing
    @test initialise!(model.momentum.p, 100000.0) === nothing
    @test initialise!(model.energy.T, Tin) === nothing

    run!(model, config; progress=false)

    inlet = boundary_average(:inlet, model.momentum.U, BCs.U, config)
    outlet = boundary_average(:outlet, model.momentum.U, BCs.U, config)
    top = boundary_average(:top, model.momentum.U, BCs.U, config)

    @test Umag ≈ inlet[1]
    @test Umag ≈ outlet[1] atol = 0.07
    @test Umag ≈ top[1] atol = 0.02

    T = model.energy.T.values
    @test all(isfinite, T)
    @test minimum(T) > Tin - 1.0
    @test maximum(T) <= Twall + 1.0
    @test maximum(T) > Tin + 0.1
    @test all(>(0), model.fluid.rho.values)
end

@testset "constant viscosity" begin
    run_variant(nu)
end
@testset "Sutherland viscosity" begin
    run_variant(Viscosity{SutherlandViscosity}(mu_ref=1.8e-5, T_ref=288.15, S=110.4))
end
