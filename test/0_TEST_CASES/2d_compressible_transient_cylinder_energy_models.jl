using XCALibre
# using CUDA # Run this if using NVIDIA GPU
# using AMDGPU # Run this if using AMD GPU

# Transient compressible CPISO solver on a cylinder at M=0.5. Runs the sensible-enthalpy
# and internal-energy formulations and checks both are physical and agree in the mean
# (they solve the same physics with a different energy variable). Also runs a
# Crank-Nicolson time-scheme case: the pressure-correction mass flux relies on the Time
# term being diagonal-only, which holds for CrankNicolson as well as Euler.

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "cylinder_d10mm_25mm.unv"
mesh_file = joinpath(grids_dir, grid)
mesh = UNV2D_mesh(mesh_file, scale=0.001)
@test typeof(mesh) <: Mesh2

workgroup = workgroupsize(mesh)
backend = CPU()
mesh_dev = adapt(backend, mesh)

gamma = 1.4
cp = 1005.0
Pr = 0.7
nu = 1e-5
R = cp*(1.0 - 1.0/gamma)
cv = cp - R
T_inf = 300.0
p_inf = 101325.0
Tref = 0.0
Mach = 0.5
U_inf = Mach*sqrt(gamma*R*T_inf)
velocity = [U_inf, 0.0, 0.0]
noflow = [0.0, 0.0, 0.0]

function run_transient_cylinder(energy, he_inlet; tscheme=Euler)
    model = Physics(
        time = Transient(),
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
                Dirichlet(:cylinder, noflow),
                Slip(:top),
                Slip(:bottom)
            ],
            p = [
                Dirichlet(:inlet, p_inf),
                Zerogradient(:outlet),
                Zerogradient(:cylinder),
                Zerogradient(:top),
                Zerogradient(:bottom)
            ],
            he = [
                he_inlet,
                Zerogradient(:outlet),
                Zerogradient(:cylinder),
                Zerogradient(:top),
                Zerogradient(:bottom)
            ]
        )
    )

    solvers = (
        U = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-7, relax=0.8, rtol=1e-2),
        p = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-7, relax=0.3, limit=(0.1*p_inf, 10*p_inf), rtol=1e-2),
        he = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-7, relax=0.5, limit=(100.0, 3000.0), rtol=1e-2)
    )

    schemes = (
        U = Schemes(divergence=Upwind, gradient=Gauss, time=tscheme),
        p = Schemes(gradient=Gauss, time=tscheme),
        he = Schemes(divergence=Upwind, gradient=Gauss, time=tscheme)
    )

    runtime = Runtime(iterations=100, write_interval=101, time_step=1e-6)
    hardware = Hardware(backend=backend, workgroup=workgroup)
    config = Configuration(; solvers, schemes, runtime, hardware, boundaries=BCs)

    GC.gc()
    @test initialise!(model.momentum.U, velocity) === nothing
    @test initialise!(model.momentum.p, p_inf) === nothing
    @test initialise!(model.energy.T, T_inf) === nothing

    residuals = run!(model, config)
    return model, residuals
end

modelSE, resSE = run_transient_cylinder(
    Energy{SensibleEnthalpy}(Tref=Tref),
    FixedTemperature(:inlet, T=T_inf, Enthalpy(cp=cp, Tref=Tref))
)
modelIE, resIE = run_transient_cylinder(
    Energy{InternalEnergy}(Tref=Tref),
    FixedTemperature(:inlet, T=T_inf, IEnergy(cv=cv, Tref=Tref))
)

@testset "physical" begin
    for (m, r) in ((modelSE, resSE), (modelIE, resIE))
        @test all(isfinite, r.Ux)
        @test all(isfinite, m.momentum.p.values)
        @test all(isfinite, m.energy.T.values)
        # Mean temperature stays near the freestream; extremes remain physical (not clamped)
        @test 295.0 < mean(m.energy.T.values) < 305.0
        @test minimum(m.energy.T.values) > 120.0
        @test maximum(m.energy.T.values) < 800.0
        @test mean(m.momentum.p.values) > 0.0
    end
end

@testset "energy models comparable" begin
    # Same physics, different energy variable: bulk fields must agree closely
    @test mean(modelSE.energy.T.values) ≈ mean(modelIE.energy.T.values) atol = 2.0
    @test mean(modelSE.momentum.p.values) ≈ mean(modelIE.momentum.p.values) rtol = 0.01
    @test mean(modelSE.momentum.U.x.values) ≈ mean(modelIE.momentum.U.x.values) rtol = 0.01
end

@testset "crank-nicolson time scheme" begin
    # Guards the mass-flux correction under Crank-Nicolson (its Time term is diagonal-only,
    # like Euler, so correct_mass_flux! must remain correct with it).
    modelCN, resCN = run_transient_cylinder(
        Energy{SensibleEnthalpy}(Tref=Tref),
        FixedTemperature(:inlet, T=T_inf, Enthalpy(cp=cp, Tref=Tref)),
        tscheme = CrankNicolson
    )
    @test all(isfinite, resCN.Ux)
    @test all(isfinite, modelCN.momentum.p.values)
    @test 295.0 < mean(modelCN.energy.T.values) < 305.0
    @test mean(modelCN.momentum.p.values) > 0.0
end
