using XCALibre
# using CUDA # Run this if using NVIDIA GPU
# using AMDGPU # Run this if using AMD GPU

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "compression_corner_2d_32_64_SF4.unv"
mesh_file = joinpath(grids_dir, grid)
mesh = UNV2D_mesh(mesh_file, scale=0.001)
@test typeof(mesh) <: Mesh2

workgroup = workgroupsize(mesh)
backend = CPU()
mesh_dev = adapt(backend, mesh)

nu = 0.0
gamma = 1.4
cp = 1005.0
R = cp*(1.0 - 1.0/gamma)
cv = cp - R
temp = 1000.0
Tref = 0.0
pressure = 100000.0
Pr = 0.7

M = 2
Umag = M*sqrt(gamma*R*temp)
velocity = [Umag, 0.0, 0.0]

# Mach 2 supersonic compression corner run with BoundedUpwind convection.
# Runs the same case with both energy models, adjusting the FixedTemperature converter.
function run_compression_corner(energy, he_inlet)
    model = Physics(
        time = Steady(),
        fluid = Fluid{Compressible}(nu=nu, cp=cp, gamma=gamma, Pr=Pr),
        turbulence = RANS{Laminar}(),
        energy = energy,
        domain = mesh_dev
        )

    BCs = assign(
        region = mesh,
        (
            U = [
                Dirichlet(:inlet, velocity),
                Zerogradient(:outlet),
                Zerogradient(:top),
                Slip(:wall)
            ],
            p = [
                Dirichlet(:inlet, pressure),
                Zerogradient(:outlet),
                Zerogradient(:top),
                Zerogradient(:wall)
            ],
            he = [
                he_inlet,
                Zerogradient(:outlet),
                Zerogradient(:top),
                Zerogradient(:wall)
            ]
        )
    )

    solvers = (
        U = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-10, relax=0.7, rtol=0.0, atol=1e-2),
        p = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-10, relax=0.3, limit=(0.5*pressure, 5*pressure), rtol=0.0, atol=1e-2),
        he = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-10, relax=0.3, limit=(800.0, 3000.0), rtol=0.0, atol=1e-2)
    )

    schemes = (
        U = Schemes(time=SteadyState, divergence=BoundedUpwind),
        p = Schemes(time=SteadyState, divergence=Upwind),
        he = Schemes(time=SteadyState, divergence=BoundedUpwind)
    )

    runtime = Runtime(iterations=1000, write_interval=1001, time_step=1)
    hardware = Hardware(backend=backend, workgroup=workgroup)
    config = Configuration(; solvers, schemes, runtime, hardware, boundaries=BCs)

    GC.gc()
    @test initialise!(model.momentum.U, velocity) === nothing
    @test initialise!(model.momentum.p, pressure) === nothing
    @test initialise!(model.energy.T, temp) === nothing

    residuals = run!(model, config)

    inlet = boundary_average(:inlet, model.momentum.U, BCs.U, config)
    outlet = boundary_average(:outlet, model.momentum.U, BCs.U, config)

    @test Umag ≈ inlet[1]                       # inlet Dirichlet enforced
    @test outlet[2] > 0.0                        # flow deflected by the oblique shock
    @test all(isfinite, residuals.Ux)            # BoundedUpwind stays bounded
    @test last(residuals.Ux) < 1e-4              # steady state reached
    @test last(residuals.p) < 1e-4
    # Oblique shock (M=2, 15°) gives T2/T1 ≈ 1.27. A physical peak guards against the
    # kinetic-energy source wrongly entering the internal-energy equation (Tmax ~2300).
    @test all(isfinite, model.energy.T.values)
    @test maximum(model.energy.T.values) < 1600.0
end

@testset "SensibleEnthalpy" begin
    run_compression_corner(
        Energy{SensibleEnthalpy}(Tref=Tref),
        FixedTemperature(:inlet, T=temp, Enthalpy(cp=cp, Tref=Tref))
    )
end

@testset "InternalEnergy" begin
    run_compression_corner(
        Energy{InternalEnergy}(Tref=Tref),
        FixedTemperature(:inlet, T=temp, IEnergy(cv=cv, Tref=Tref))
    )
end
