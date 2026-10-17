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

function make_bcs(mesh_dev; slip=Slip)
    assign(
        region=mesh_dev,
        (
            U = [
                Dirichlet(:inlet,    velocity),
                Zerogradient(:outlet),
                Dirichlet(:cylinder, noflow),
                slip(:top),
                slip(:bottom)
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
function run_godunov_test(flux, reconstruction, time_stepping, adaptive; gradient=Gauss)
    model = make_model(mesh_dev)
    BCs   = make_bcs(mesh_dev)

    runtime = if adaptive === nothing
        Runtime(iterations=30, write_interval=-1, time_step=1e-7)
    else
        Runtime(iterations=30, write_interval=-1, time_step=1e-7,
                adaptive=adaptive)
    end

    schemes = (
        U             = Schemes(gradient=gradient),
        p             = Schemes(gradient=gradient),
        T             = Schemes(gradient=gradient),
        flux          = flux,
        time_stepping = time_stepping,
    )
    # `nothing` leaves `reconstruction` out, which selects the default (Upwind)
    isnothing(reconstruction) || (schemes = merge(schemes, (; reconstruction)))

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
        ("Rusanov+Upwind",          Rusanov(), Upwind(),          FEuler(), nothing, Gauss),
        ("HLLC+Upwind",             HLLC(),    Upwind(),          FEuler(), nothing, Gauss),
        ("HLLC+default recon",      HLLC(),    nothing,           FEuler(), nothing, Gauss),
        ("HLLC+MUSCL{VanLeer}",     HLLC(),    MUSCL{VanLeer}(),  FEuler(), nothing, Gauss),
        ("HLLC+MUSCL{MinMod}",      HLLC(),    MUSCL{MinMod}(),   FEuler(), nothing, Gauss),
        ("HLLC+MUSCL{Superbee}",    HLLC(),    MUSCL{Superbee}(), FEuler(), nothing, Gauss),
        ("HLLC+MUSCL+LeastSquares", HLLC(),    MUSCL{VanLeer}(),  FEuler(), nothing, LeastSquares),
        ("HLLC+MUSCL+RK2",          HLLC(),    MUSCL{VanLeer}(),  RK2(),    nothing, Gauss),
        ("HLLC+MUSCL+adaptive",     HLLC(),    MUSCL{VanLeer}(),  FEuler(), AdaptiveTimeStepping(maxCo=0.3), Gauss),
    ]
    for (label, flux, recon, ts, adaptive, gradient) in cases
        @testset "$label" begin
            run_godunov_test(flux, recon, ts, adaptive; gradient)
        end
    end
end

# ── Local (per-cell) time stepping and multistage Runge-Kutta ───────────────
# FEuler and RK2 with local time stepping run in the turbulence_dt and far-field sets below
@testset "Local time stepping and multistage RK" begin
    for (label, ts) in [("RK3", RK3()), ("RK5", RK5())]
        @testset "LTS + $label" begin
            run_godunov_test(
                HLLC(), Upwind(), ts,
                LocalTimeStepping(maxCo=0.5, startCo=0.1, ramp=20)
            )
        end
    end

    @testset "turbulence_dt option" begin
        for rule in (:min, :mean, :max)
            lts = LocalTimeStepping(maxCo=0.5, turbulence_dt=rule)
            @test isbitstype(typeof(lts))   # needed to pass it to GPU kernels
            run_godunov_test(HLLC(), Upwind(), FEuler(), lts)
        end
        @test_throws ArgumentError LocalTimeStepping(turbulence_dt=:median)
    end

    @testset "residuals and force history" begin
        model = make_model(mesh_dev)
        schemes = (
            U = Schemes(gradient=Gauss), p = Schemes(gradient=Gauss), T = Schemes(gradient=Gauss),
            flux = HLLC(), time_stepping = RK3(), reconstruction = Upwind(),
        )
        runtime = Runtime(iterations=40, write_interval=-1, time_step=1e-7,
                          adaptive=LocalTimeStepping(maxCo=0.5))
        config = Configuration(solvers=solvers, schemes=schemes, runtime=runtime,
                               hardware=hardware, boundaries=make_bcs(mesh_dev))
        initialise!(model.momentum.U, velocity)
        initialise!(model.momentum.p, p_inf)
        initialise!(model.energy.T,   T_inf)
        res = run!(model, config; progress=false,
                   forces=(patches=[:cylinder], interval=10))
        for k in (:rho, :rhoU, :rhoE, :rho_rel, :rhoU_rel, :rhoE_rel)
            @test haskey(res, k)
            @test length(res[k]) == 40
            @test all(isfinite, res[k])
        end
        @test res.rho_rel[1] > 0 && res.rho[1] > 0
        @test res.forces.cylinder.iteration == [10, 20, 30, 40]
        @test all(f -> all(isfinite, f), res.forces.cylinder.pressure)
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

# ── Characteristic far field (U, p, T all Dirichlet), subsonic free stream ──
@testset "Characteristic far field" begin
    vel = [0.6 * a_inf, 0.0, 0.0]
    model = make_model(mesh_dev)
    BCs = assign(region=mesh_dev, (
        U = [Dirichlet(:inlet, vel), Dirichlet(:outlet, vel), Slip(:cylinder),
             Dirichlet(:top, vel), Dirichlet(:bottom, vel)],
        p = [Dirichlet(:inlet, p_inf), Dirichlet(:outlet, p_inf), Zerogradient(:cylinder),
             Dirichlet(:top, p_inf), Dirichlet(:bottom, p_inf)],
        T = [Dirichlet(:inlet, T_inf), Dirichlet(:outlet, T_inf), Zerogradient(:cylinder),
             Dirichlet(:top, T_inf), Dirichlet(:bottom, T_inf)],
        nut = [Extrapolated(:inlet), Extrapolated(:outlet), Zerogradient(:cylinder),
               Extrapolated(:top), Extrapolated(:bottom)]))
    schemes = (U=Schemes(gradient=Gauss), p=Schemes(gradient=Gauss), T=Schemes(gradient=Gauss),
               flux=HLLC(), time_stepping=RK2(), reconstruction=Upwind())
    runtime = Runtime(iterations=300, write_interval=-1, time_step=1e-7,
                      adaptive=LocalTimeStepping(maxCo=0.5, startCo=0.1, ramp=50))
    config = Configuration(solvers=solvers, schemes=schemes, runtime=runtime,
                           hardware=hardware, boundaries=BCs)
    initialise!(model.momentum.U, vel)
    initialise!(model.momentum.p, p_inf)
    initialise!(model.energy.T, T_inf)
    res = run!(model, config; progress=false)
    @test all(isfinite, res.rho)
    @test res.rho[end] < res.rho[1]
    p = model.momentum.p.values
    @test 0.2 * p_inf < minimum(p) && maximum(p) < 2.5 * p_inf
end

# ── Force history on Slip / Symmetry patches ────────────────────────────────
@testset "Viscous force on Slip and Symmetry patches" begin
    for (label, SlipBC) in [("Slip", Slip), ("Symmetry", Symmetry)]
        @testset "$label" begin
            model = make_model(mesh_dev)
            BCs = make_bcs(mesh_dev; slip=SlipBC)
            schemes = (U=Schemes(), p=Schemes(), T=Schemes(), flux=HLLC(),
                       time_stepping=FEuler(), reconstruction=Upwind())
            runtime = Runtime(iterations=10, write_interval=-1, time_step=1e-7)
            config = Configuration(solvers=solvers, schemes=schemes, runtime=runtime,
                                   hardware=hardware, boundaries=BCs)
            initialise!(model.momentum.U, velocity)
            initialise!(model.momentum.p, p_inf)
            initialise!(model.energy.T, T_inf)
            res = run!(model, config; progress=false,
                       forces=(patches=[:cylinder, :top], interval=5))
            @test res.forces.top.iteration == [5, 10]
            @test all(f -> all(iszero, f), res.forces.top.viscous) # no shear on a slip patch
            @test all(f -> all(isfinite, f) && f[2] != 0, res.forces.top.pressure) # pressure force kept
            @test all(f -> all(isfinite, f), res.forces.cylinder.viscous)
            @test viscous_force(:top, model, config) == zeros(3)
            @test pressure_force(:top, model, config) == res.forces.top.pressure[end]
        end
    end
end
