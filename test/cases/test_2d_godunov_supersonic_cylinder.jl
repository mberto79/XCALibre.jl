using XCALibre

# ── mesh (shared across all sub-tests) ─────────────────────────────────────
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh_file = joinpath(grids_dir, "cylinder_d10mm_5mm.unv")
mesh = UNV2D_mesh(mesh_file, scale=0.001)
@test typeof(mesh) <: Mesh2

backend   = CPU()
workgroup = workgroupsize(mesh)
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
        Runtime(iterations=30, write_interval=-1, time_step=1e-7)
    else
        Runtime(iterations=30, write_interval=-1, time_step=1e-7,
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

    residuals = run!(model, config; progress=false)

    # Basic sanity: residuals returned, all finite
    @test residuals isa NamedTuple
    @test haskey(residuals, :rho)
    @test all(isfinite, residuals.rho)

    rho_inf = p_inf/(R_gas*T_inf)
    rho, p, T = model.fluid.rho.values, model.momentum.p.values, model.energy.T.values
    @test all(>(0), rho) && all(>(0), p) && all(>(0), T)
    @test maximum(p) < 6*p_inf # impulsive-start overshoot (observed ~4.2)
    @test maximum(rho) < 2.8*rho_inf
end

# ── Unique scheme combinations ─────────────────────────────────────────────
@testset "Scheme combinations" begin
    cases = [
        ("Rusanov+Upwind",       Rusanov(), Upwind(),          FEuler(), nothing),
        ("HLLC+Upwind",          HLLC(),    Upwind(),          FEuler(), nothing),
        ("HLLC+MUSCL{VanLeer}",  HLLC(),    MUSCL{VanLeer}(),  FEuler(), nothing),
        ("HLLC+MUSCL{MinMod}",   HLLC(),    MUSCL{MinMod}(),   FEuler(), nothing),
        ("HLLC+MUSCL{Superbee}", HLLC(),    MUSCL{Superbee}(), FEuler(), nothing),
        ("HLLC+MUSCL+RK2",       HLLC(),    MUSCL{VanLeer}(),  RK2(),    nothing),
        ("HLLC+MUSCL+adaptive",  HLLC(),    MUSCL{VanLeer}(),  FEuler(), AdaptiveTimeStepping(maxCo=0.3)),
    ]
    for (label, flux, recon, ts, adaptive) in cases
        @testset "$label" begin
            run_godunov_test(flux, recon, ts, adaptive)
        end
    end
end

# ── Robin BCs are rejected ──────────────────────────────────────────────────
@testset "Robin BC unsupported" begin
    BCs = assign(
        region=mesh_dev,
        (
            U = [Dirichlet(:inlet, velocity), Zerogradient(:outlet), Dirichlet(:cylinder, noflow), Slip(:top), Slip(:bottom)],
            p = [Dirichlet(:inlet, p_inf), Robin(:outlet, a=0.0, b=1.0), Zerogradient(:cylinder), Zerogradient(:top), Zerogradient(:bottom)],
            T = [Dirichlet(:inlet, T_inf), Zerogradient(:outlet), Zerogradient(:cylinder), Zerogradient(:top), Zerogradient(:bottom)],
            nut = [Extrapolated(:inlet), Extrapolated(:outlet), Zerogradient(:cylinder), Symmetry(:top), Symmetry(:bottom)]
        )
    )
    config = Configuration(
        solvers=solvers, schemes=(U=Schemes(), p=Schemes(), T=Schemes()),
        runtime=Runtime(iterations=1, write_interval=-1, time_step=1e-7),
        hardware=hardware, boundaries=BCs
    )
    @test_throws "Robin boundary conditions on `p`" run!(make_model(mesh_dev), config)
end
