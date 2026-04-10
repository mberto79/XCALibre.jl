using XCALibre

# ── mesh (shared across all sub-tests) ─────────────────────────────────────
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh_file = joinpath(grids_dir, "cylinder_d10mm_5mm.unv")
mesh = UNV2D_mesh(mesh_file, scale=0.001)
@test typeof(mesh) <: Mesh2

backend   = CPU()
workgroup = AutoTune()
hardware  = Hardware(backend=backend, workgroup=workgroup)
mesh_dev  = adapt(backend, mesh)

# ── freestream conditions at M=1.2 ─────────────────────────────────────────
gamma = 1.4
cp    = 1005.0
Pr    = 0.7
nu    = 1e-5
T_inf = 300.0
p_inf = 101325.0
R_gas = cp * (1.0 - 1.0/gamma)
a_inf = sqrt(gamma * R_gas * T_inf)
Mach  = 1.2
U_inf = Mach * a_inf

velocity = [U_inf, 0.0, 0.0]
noflow   = [0.0,   0.0, 0.0]

# ── helper to build a fresh model & BCs ────────────────────────────────────
function make_model(mesh_dev)
    Physics(
        time      = Transient(),
        fluid     = Fluid{SupersonicFlow}(nu=nu, cp=cp, gamma=gamma, Pr=Pr),
        turbulence = LES{Smagorinsky}(),
        energy    = Energy{SensibleEnthalpy}(Tref=0.0),
        domain    = mesh_dev
    )
end

function make_bcs(mesh_dev)
    assign(
        region=mesh_dev,
        (
            U = [
                Dirichlet(:inlet,    velocity),
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
            T = [
                Dirichlet(:inlet, T_inf),
                Zerogradient(:outlet),
                Zerogradient(:cylinder),
                Zerogradient(:top),
                Zerogradient(:bottom)
            ],
            nut = [
                Extrapolated(:inlet),
                Extrapolated(:outlet),
                Zerogradient(:cylinder),
                Symmetry(:top),
                Symmetry(:bottom)
            ]
        )
    )
end

solvers = (rho = (convergence = 1e-15,),)

# ── sub-test runner ─────────────────────────────────────────────────────────
function run_godunov_test(flux, reconstruction, time_stepping, adaptive)
    model = make_model(mesh_dev)
    BCs   = make_bcs(mesh_dev)

    runtime = if adaptive === nothing
        Runtime(iterations=100, write_interval=-1, time_step=1e-7)
    else
        Runtime(iterations=100, write_interval=-1, time_step=1e-7,
                adaptive=adaptive)
    end

    schemes = (
        U             = Schemes(gradient=Gauss),
        p             = Schemes(gradient=Gauss),
        T             = Schemes(gradient=Gauss),
        flux          = flux,
        time_stepping = time_stepping,
        reconstruction = reconstruction
    )

    config = Configuration(
        solvers=solvers, schemes=schemes, runtime=runtime,
        hardware=hardware, boundaries=BCs
    )

    GC.gc(true)
    initialise!(model.momentum.U, velocity)
    initialise!(model.momentum.p, p_inf)
    initialise!(model.energy.T,   T_inf)

    residuals = run!(model, config)

    # Basic sanity: residuals returned, all finite
    @test residuals isa NamedTuple
    @test haskey(residuals, :rho)
    @test all(isfinite, residuals.rho)
end

# ── Flux schemes ────────────────────────────────────────────────────────────
@testset "Flux schemes" begin
    for flux in [Rusanov(), HLLC()]
        @testset "flux=$(typeof(flux))" begin
            run_godunov_test(flux, Upwind(), FEuler(), nothing)
        end
    end
end

# ── Reconstruction schemes ──────────────────────────────────────────────────
@testset "Reconstruction schemes" begin
    reconstructions = [
        ("Upwind",          Upwind()),
        ("MUSCL{VanLeer}",  MUSCL{VanLeer}()),
        ("MUSCL{MinMod}",   MUSCL{MinMod}()),
        ("MUSCL{Superbee}", MUSCL{Superbee}()),
    ]
    for (label, recon) in reconstructions
        @testset "reconstruction=$label" begin
            run_godunov_test(HLLC(), recon, FEuler(), nothing)
        end
    end
end

# ── Time stepping schemes ───────────────────────────────────────────────────
@testset "Time stepping schemes" begin
    for (label, ts) in [("FEuler", FEuler()), ("RK2", RK2())]
        @testset "time_stepping=$label" begin
            run_godunov_test(HLLC(), MUSCL{VanLeer}(), ts, nothing)
        end
    end
end

# ── Adaptive vs fixed dt ────────────────────────────────────────────────────
@testset "Adaptive time stepping" begin
    @testset "fixed dt" begin
        run_godunov_test(HLLC(), MUSCL{VanLeer}(), FEuler(), nothing)
    end
    for maxCo in [0.3, 0.5]
        @testset "adaptive maxCo=$maxCo" begin
            run_godunov_test(
                HLLC(), MUSCL{VanLeer}(), FEuler(),
                AdaptiveTimeStepping(maxCo=maxCo)
            )
        end
    end
end
