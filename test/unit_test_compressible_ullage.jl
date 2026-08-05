# Validation of the compressible (ideal-gas) ullage — step 5 of the LH2 tank work.
#
# The central test has an EXACT analytical target. For a sealed rigid volume of
# ideal gas heated at total rate Q, the first law gives
#
#     m*cv*dT/dt = Q,        p = (m/V)*R*T   =>   dp/dt = R*Q/(V*cv)
#
# with cv = cp - R. Our temperature equation solves rho*cp*DT/Dt, so reaching
# that result depends entirely on the pressure-work source beta*T*Dp/Dt (which is
# exactly Dp/Dt for an ideal gas, since beta = 1/T): integrating over the sealed
# volume,
#
#     m*cp*dT/dt = Q + V*dp/dt = Q + m*R*dT/dt   =>   m*(cp - R)*dT/dt = Q
#
# So this test discriminates sharply: WITHOUT the pressure-work term the answer
# would come out as R*Q/(V*cp), low by a factor cp/cv ~ 1.5. Both predictions are
# asserted against, so a regression that drops the term cannot pass.

using XCALibre
using Test

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "quad40.unv"), scale=0.001)

backend = CPU(); workgroup = AutoTune()
hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

noSlip = [0.0, 0.0, 0.0]

M_H2   = 2.01588e-3                 # [kg/mol]
R_H2   = XCALibre.ModelPhysics.R_UNIVERSAL/M_H2   # ~4124.2 J/kg/K
CP_VAP = 12200.0                    # [J/kg/K]
CV_VAP = CP_VAP - R_H2              # ideal gas: cv = cp - R
T0     = 20.43
P0     = 103.0e3

# Zero gravity keeps the pressure spatially uniform, so the lumped balance above
# applies without a hydrostatic correction.
NO_GRAVITY = Gravity([0.0, 0.0, 0.0])

patch_area(m, name) = sum(m.faces[f].area
    for f in only(b for b in m.boundaries if b.name === name).IDs_range)

domain_volume(m) = sum(c.volume for c in m.cells)

A_WALL = patch_area(mesh_dev, :bottom)
VOLUME = domain_volume(mesh_dev)

"""Sealed box, all vapour (alpha = 0), heat flux q on :bottom, adiabatic elsewhere."""
function build(q; iterations, dt, liquid_frac=0.0, pressure_form=nothing)
    # `pressure_form = nothing` omits the keyword entirely, so the default path is
    # exercised as a user would actually hit it rather than by passing :volume.
    extra = pressure_form === nothing ? (;) : (pressure_form=pressure_form,)
    model = Physics(
        time = Transient(),
        fluid = Fluid{Multiphase}(;
            model = VOF(cAlpha=1.0, sigma=0.0),
            phases = (
                Phase(rho=70.8, mu=13.2e-6, k=0.100, cp=9660.0, beta=0.0164),
                Phase(rho=IdealGas(M=M_H2), mu=1.11e-6, k=0.0169, cp=CP_VAP),
            ),
            p_operating = P0,
            gravity = NO_GRAVITY,
            extra...
        ),
        turbulence = RANS{Laminar}(),
        energy = Energy{TwoPhaseTemperature}(Tref=T0),
        domain = mesh_dev
    )

    # No pressure-defining boundary anywhere: the level is set by the
    # compressibility term. That is the whole point of a sealed tank.
    BCs = assign(
        region = mesh_dev,
        (
            U = [Wall(:inlet, noSlip), Wall(:outlet, noSlip),
                 Wall(:bottom, noSlip), Wall(:top, noSlip)],
            p_rgh = [Zerogradient(:inlet), Zerogradient(:outlet),
                     Zerogradient(:bottom), Zerogradient(:top)],
            alpha = [Zerogradient(:inlet), Zerogradient(:outlet),
                     Zerogradient(:bottom), Zerogradient(:top)],
            T = [Zerogradient(:inlet), Zerogradient(:outlet),
                 FixedHeatFlux(:bottom, q), Zerogradient(:top)],
        )
    )

    sch = Schemes(time=Euler, divergence=Upwind, laplacian=Linear, gradient=Gauss)
    schemes = (U=sch, p=sch, p_rgh=sch, alpha=sch, T=sch)

    bicg() = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(),
                         convergence=1e-10, relax=1.0, rtol=0.0, atol=1e-14)
    solvers = (U=bicg(),
               p_rgh=SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(),
                                 convergence=1e-10, relax=1.0, rtol=0.0, atol=1e-14),
               alpha=bicg(), T=bicg())

    runtime = Runtime(iterations=iterations, time_step=dt, write_interval=-1)
    config = Configuration(solvers=solvers, schemes=schemes,
                           runtime=runtime, hardware=hardware, boundaries=BCs)

    initialise!(model.momentum.U, noSlip)
    initialise!(model.fluid.p_rgh, 0.0)
    initialise!(model.fluid.alpha, liquid_frac)
    initialise!(model.energy.T, T0)

    return model, config
end

mean_pressure(model) = sum(model.momentum.p.values)/length(model.momentum.p.values)
vapour_mass(model) = sum(model.fluid.phases[2].rho.values[i]*model.domain.cells[i].volume
                         for i in eachindex(model.domain.cells))

@testset "IdealGas: construction and compressibility" begin
    g = IdealGas(M=M_H2)
    @test g.R ≈ R_H2 rtol=1e-12
    @test IdealGas(R=4124.2).R == 4124.2
    @test_throws ArgumentError IdealGas()
    @test_throws ArgumentError IdealGas(R=1.0, M=1.0)

    # ideal gas: (1/rho)(drho/dp) = 1/p exactly; constant density: 0
    @test phase_compressibility(g, 1.0e5, 20.0) ≈ 1/1.0e5
    @test phase_compressibility(ConstEos(70.8), 1.0e5, 20.0) == 0.0

    # beta*T = 1 for an ideal gas (beta = 1/T)
    @test phase_betaT(g, 0.0, 20.43) == 1.0
    @test phase_betaT(ConstEos(70.8), 0.0164, 20.43) ≈ 0.0164*20.43
end

@testset "Compressible ullage requires p_operating and a temperature field" begin
    # An ideal gas at zero absolute pressure has zero density, so the datum is
    # mandatory rather than optional.
    bad = Physics(
        time = Transient(),
        fluid = Fluid{Multiphase}(
            model = VOF(cAlpha=1.0, sigma=0.0),
            phases = (Phase(rho=70.8, mu=13.2e-6, k=0.1, cp=9660.0),
                      Phase(rho=IdealGas(M=M_H2), mu=1.11e-6, k=0.0169, cp=CP_VAP)),
            gravity = NO_GRAVITY),          # no p_operating
        turbulence = RANS{Laminar}(),
        energy = Energy{TwoPhaseTemperature}(Tref=T0),
        domain = mesh_dev)

    _, cfg = build(0.0, iterations=1, dt=1e-3)
    err = try; run!(bad, cfg); nothing; catch e; e; end
    @test err isa ArgumentError
    @test occursin("p_operating", err.msg)
end

@testset "Ideal gas density follows p/(R*T) per cell" begin
    model, config = build(0.0, iterations=5, dt=1e-3)
    run!(model, config)

    rho_v = model.fluid.phases[2].rho.values
    T = model.energy.T.values
    p = model.momentum.p.values

    for i in eachindex(rho_v)
        @test rho_v[i] ≈ p[i]/(R_H2*T[i]) rtol=1e-10
    end
end

@testset "Sealed rigid volume: dp/dt = R*Q/(V*cv)  [exact]" begin
    q = 1000.0                        # [W/m^2]
    Q = q*A_WALL                      # [W]
    nsteps, dt = 200, 1.0e-3

    # settle one step, then measure the increment
    model, cfg1 = build(q, iterations=1, dt=dt)
    run!(model, cfg1)
    p0 = mean_pressure(model)
    m0 = vapour_mass(model)

    _, cfgN = build(q, iterations=nsteps, dt=dt)
    run!(model, cfgN)
    p1 = mean_pressure(model)
    m1 = vapour_mass(model)

    measured = (p1 - p0)/(nsteps*dt)
    expected_cv = R_H2*Q/(VOLUME*CV_VAP)      # correct: pressure work included
    expected_cp = R_H2*Q/(VOLUME*CP_VAP)      # wrong: pressure work omitted

    @info "sealed ullage pressurisation" A_WALL VOLUME Q measured expected_cv expected_cp ratio=measured/expected_cv

    @test measured > 0                                       # it must pressurise
    @test isapprox(measured, expected_cv; rtol=0.02)          # matches cv form
    @test !isapprox(measured, expected_cp; rtol=0.10)         # NOT the cp form

    # Sealed tank: the vapour mass should be constant, since for a rigid closed
    # volume p/T = mR/V is invariant.
    #
    # In practice it drifts slowly, and the drift is NOT bounded — it grows with
    # the accumulated pressure change. It traces directly to the ~0.1% error in
    # dp/dt above: a slightly low pressure rate lets p/T creep. So this asserts a
    # magnitude bound, not invariance, and the bound scales with how far the tank
    # has pressurised rather than being a fixed number.
    #
    # For the full K-Site run (O(10^6) steps, pressure doubling) this is worth
    # monitoring: see the note in dev_notes_LH2_implementation_plan.md.
    drift = abs(m1 - m0)/abs(m0)
    p_change = (p1 - p0)/p0
    @info "sealed vapour mass drift" m0 m1 drift p_change ratio=drift/p_change

    @test drift < 1e-4
    # The drift must stay a small fraction of the relative pressure change, i.e.
    # it is a consistency error in p/T, not an independent mass leak.
    @test drift < 0.05*p_change
end

@testset "No heating: no pressurisation" begin
    # With q = 0 a sealed tank must hold its pressure. This also confirms the
    # compressibility term alone does not manufacture a drift.
    model, config = build(0.0, iterations=100, dt=1.0e-3)
    run!(model, config)
    @test isapprox(mean_pressure(model), P0; rtol=1e-9)
    @test maximum(abs, model.energy.T.values .- T0) < 1e-6
end

# =============================================================================
#  Mass form of the pressure equation (`pressure_form = :mass`)
# =============================================================================
#
#  The mass form solves d(rho_m)/dt + div(rho_m u) = 0 instead of the low-Mach
#  volume constraint. It is the same statement multiplied through by rho_m, so it
#  must reproduce the same PHYSICS — and the sealed-tank test above has an exact
#  analytical answer that does not care how the equation was scaled. That makes
#  it the right acceptance test: if the conversion dropped or double-counted a
#  density anywhere, dp/dt lands on the wrong number rather than merely drifting.
#
#  These tests deliberately reuse the exact targets rather than comparing the two
#  forms to each other, which would pass even if both were wrong.

@testset "Mass form reproduces the exact sealed-tank pressurisation" begin
    q = 1000.0
    Q = q*A_WALL
    nsteps, dt = 200, 1.0e-3

    model, cfg1 = build(q, iterations=1, dt=dt, pressure_form=:mass)
    run!(model, cfg1)
    p0 = mean_pressure(model)
    m0 = vapour_mass(model)

    _, cfgN = build(q, iterations=nsteps, dt=dt, pressure_form=:mass)
    run!(model, cfgN)
    p1 = mean_pressure(model)
    m1 = vapour_mass(model)

    measured = (p1 - p0)/(nsteps*dt)
    expected_cv = R_H2*Q/(VOLUME*CV_VAP)
    expected_cp = R_H2*Q/(VOLUME*CP_VAP)

    @info "mass-form sealed ullage pressurisation" measured expected_cv ratio=measured/expected_cv

    @test measured > 0
    @test isapprox(measured, expected_cv; rtol=0.02)
    @test !isapprox(measured, expected_cp; rtol=0.10)

    # Same bounds as the volume form. The mass form enforces mass conservation
    # directly, so this drift is the quantity it should most affect - but the
    # assertion is the independent bound, not "better than the other form".
    drift = abs(m1 - m0)/abs(m0)
    p_change = (p1 - p0)/p0
    @info "mass-form sealed vapour mass drift" drift p_change ratio=drift/p_change
    @test drift < 1e-4
    @test drift < 0.05*p_change
end

@testset "Mass and volume forms agree" begin
    # The two forms must land on the same solution to well within the ~1% by
    # which either misses the analytical target, since they are the same
    # continuum statement.
    #
    # They do NOT agree to solver precision, and should not be expected to: the
    # scaling is not uniform across rows. `rho_f` varies over the mesh, and the
    # time coefficient is sum_i alpha_i rho_i psi_i rather than
    # rho_m*sum_i alpha_i psi_i. So this bounds the difference at the
    # discretisation level (1e-5 relative), not at machine level. A tighter
    # assertion here would be asserting something false.
    q, nsteps, dt = 1000.0, 50, 1.0e-3

    mv, cv = build(q, iterations=nsteps, dt=dt)
    run!(mv, cv)

    mm, cm = build(q, iterations=nsteps, dt=dt, pressure_form=:mass)
    run!(mm, cm)

    @test isapprox(mean_pressure(mv), mean_pressure(mm); rtol=1e-6)
    @test maximum(abs, mv.energy.T.values .- mm.energy.T.values) < 1e-5*T0
end

@testset "Mass form runs two-phase" begin
    # alpha = 0.5 is where the two forms genuinely differ: the compressibility
    # coefficient is sum_i alpha_i rho_i psi_i, not rho_m*sum_i alpha_i psi_i,
    # and with rho_l/rho_v ~ 700 here the two are far apart. No exact target, so
    # this asserts only that the path is stable and physical.
    model, config = build(1000.0, iterations=50, dt=1.0e-3,
                          liquid_frac=0.5, pressure_form=:mass)
    run!(model, config)

    @test all(isfinite, model.momentum.p.values)
    @test all(isfinite, model.energy.T.values)
    @test mean_pressure(model) > P0            # heating a sealed tank raises p
    @test minimum(model.energy.T.values) > T0 - 1e-6   # nothing cools below start
end

@testset "pressure_form is validated" begin
    model, config = build(1000.0, iterations=1, dt=1e-3, pressure_form=:volumetric)
    err = try; run!(model, config); nothing; catch e; e; end
    @test err isa ArgumentError
    @test occursin("pressure_form", err.msg)
end

@testset "pref is refused for a compressible run" begin
    # Pinning a reference cell would suppress exactly the pressure rise being
    # solved for, so it must be an error rather than a silent no-op.
    model, config = build(1000.0, iterations=1, dt=1e-3)
    err = try; run!(model, config; pref=0.0); nothing; catch e; e; end
    @test err isa ArgumentError
    @test occursin("pref", err.msg)
end
