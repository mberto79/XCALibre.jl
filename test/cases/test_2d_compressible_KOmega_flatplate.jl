using XCALibre

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "flatplate_2D_laminar.unv"), scale=0.001)
@test typeof(mesh) <: Mesh2

workgroup = 1023
backend = CPU()
mesh_dev = adapt(backend, mesh)

Umag = 10
velocity = [Umag, 0.0, 0.0]
nu = 1e-4
cp = 1005.0
gamma = 1.4
Pr = 0.7
k_inlet = 0.375
ω_inlet = 1000
Tin = 300.0
Twall = 310.0

# Same mesh and setup for every variant; only viscosity law and wall thermal BC change
function run_variant(; visc, wall_he, p_solver, iterations=30)
    model = Physics(
        time = Steady(),
        fluid = Fluid{WeaklyCompressible}(nu=visc, cp=cp, gamma=gamma, Pr=Pr),
        turbulence = RANS{KOmega}(),
        energy = Energy{SensibleEnthalpy}(Tref=288.15),
        domain = mesh_dev
    )
    BCs = assign(
        region=mesh_dev,
        (
            U = [
                Dirichlet(:inlet, velocity),
                Zerogradient(:outlet),
                Wall(:wall, [0.0, 0.0, 0.0]),
                Extrapolated(:top)
            ],
            p = [
                Zerogradient(:inlet),
                Dirichlet(:outlet, 100000.0),
                Wall(:wall),
                Extrapolated(:top)
            ],
            he = [
                FixedTemperature(:inlet, T=Tin, Enthalpy(cp=cp, Tref=288.15)),
                Zerogradient(:outlet),
                wall_he,
                Extrapolated(:top)
            ],
            k = [
                Dirichlet(:inlet, k_inlet),
                Extrapolated(:outlet),
                Dirichlet(:wall, 0.0),
                Extrapolated(:top)
            ],
            omega = [
                Dirichlet(:inlet, ω_inlet),
                Extrapolated(:outlet),
                OmegaWallFunction(:wall),
                Extrapolated(:top)
            ],
            nut = [
                Dirichlet(:wall, 0.0),
                Extrapolated.([:inlet, :outlet, :top])...
            ]
        )
    )
    solvers = (
        U = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-7, relax=0.7, rtol=1e-1),
        p = SolverSetup(solver=p_solver, preconditioner=DILU(), convergence=1e-7, relax=0.2, rtol=1e-2),
        he = SolverSetup(solver=Bicgstab(), preconditioner=DILU(), convergence=1e-7, relax=0.7, rtol=1e-1),
        k = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-7, relax=0.7, rtol=1e-1),
        omega = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-7, relax=0.7, rtol=1e-1)
    )
    schemes = (
        U = Schemes(divergence=Upwind, limiter=nothing),
        p = Schemes(divergence=Upwind, limiter=nothing),
        he = Schemes(divergence=Upwind),
        k = Schemes(divergence=Upwind),
        omega = Schemes(divergence=Upwind)
    )
    runtime = Runtime(iterations=iterations, write_interval=iterations, time_step=1)
    hardware = Hardware(backend=backend, workgroup=workgroup)
    config = Configuration(
        solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

    GC.gc()

    @test initialise!(model.momentum.U, velocity) === nothing
    @test initialise!(model.momentum.p, 100000.0) === nothing
    @test initialise!(model.energy.T, Tin) === nothing
    @test initialise!(model.turbulence.k, k_inlet) === nothing
    @test initialise!(model.turbulence.omega, ω_inlet) === nothing
    @test initialise!(model.turbulence.nut, k_inlet/ω_inlet) === nothing

    run!(model, config; progress=false)
    return model, config, BCs
end

function check_variant(model, config, BCs; heatflux=false)
    inlet = boundary_average(:inlet, model.momentum.U, BCs.U, config)
    outlet = boundary_average(:outlet, model.momentum.U, BCs.U, config)
    top = boundary_average(:top, model.momentum.U, BCs.U, config)
    @test Umag ≈ inlet[1]
    @test Umag ≈ outlet[1] atol = 0.95
    @test Umag ≈ top[1] atol = 0.15

    T = model.energy.T.values
    @test all(isfinite, T)
    @test minimum(T) > Tin - 1.0
    heatflux ? (@test maximum(T) > Tin + 0.1) : (@test maximum(T) <= Twall + 1.0)
    @test all(>(0), model.turbulence.k.values)
    @test all(>(0), model.turbulence.omega.values)
    @test all(>=(0), model.turbulence.nut.values)
    @test all(>(0), model.fluid.rho.values)
end

enth = Enthalpy(cp=cp, Tref=288.15)
sutherland = Viscosity{SutherlandViscosity}(mu_ref=1.8e-5, T_ref=288.15, S=110.4)

@testset "fixedT" begin
    check_variant(run_variant(visc=nu, wall_he=FixedTemperature(:wall, T=Twall, enth), p_solver=Cg())...)
end
@testset "fixedHeatFlux" begin
    check_variant(run_variant(visc=nu, wall_he=FixedHeatFlux(:wall, 10000.0), p_solver=Bicgstab())...; heatflux=true)
end
@testset "ConstantViscosity" begin
    check_variant(run_variant(visc=Viscosity{ConstantViscosity}(nu=nu), wall_he=FixedTemperature(:wall, T=Twall, enth), p_solver=Cg())...)
end
@testset "SutherlandViscosity" begin
    check_variant(run_variant(visc=sutherland, wall_he=FixedTemperature(:wall, T=Twall, enth), p_solver=Cg())...)
end
