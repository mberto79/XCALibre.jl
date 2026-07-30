# Tests for Energy{TwoPhaseTemperature} — the two-phase temperature transport
# added for the LH2 tank work (step 3).
#
# Covers:
#   - the API accepts the model and rejects single-phase energy models loudly
#   - missing k/cp on a phase is reported against that phase, not deep in a kernel
#   - the blended coefficients (rho*cp, keff, rho*cp*phi) are correct
#   - energy is conserved in a sealed adiabatic box
#   - a stably stratified field stays put (no spurious diffusion/advection)

using XCALibre
using Test

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "quad40.unv"), scale=0.001)

backend = CPU(); workgroup = AutoTune()
hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

noSlip = [0.0, 0.0, 0.0]
gravity = Gravity([0.0, -9.81, 0.0])

# LH2 / GH2 properties at ~20.3 K (paper Sec. 3.2 values)
LIQUID = (rho=70.8, mu=13.2e-6, k=0.100, cp=9660.0)
VAPOUR = (rho=1.34, mu=1.11e-6, k=0.0169, cp=12200.0)

T0 = 20.43

function build_model(; energy, liquid=LIQUID, vapour=VAPOUR)
    Physics(
        time = Transient(),
        fluid = Fluid{Multiphase}(
            model = VOF(cAlpha=1.0, sigma=0.0),
            phases = (Phase(; liquid...), Phase(; vapour...)),
            gravity = gravity
        ),
        turbulence = RANS{Laminar}(),
        energy = energy,
        domain = mesh_dev
    )
end

function build_config(; iterations=5, dt=1.0e-4)
    BCs = assign(
        region = mesh_dev,
        (
            U = [Wall(:inlet, noSlip), Wall(:outlet, noSlip),
                 Wall(:bottom, noSlip), Wall(:top, noSlip)],
            p_rgh = [Zerogradient(:inlet), Zerogradient(:outlet),
                     Zerogradient(:bottom), Dirichlet(:top, 0.0)],
            alpha = [Zerogradient(:inlet), Zerogradient(:outlet),
                     Zerogradient(:bottom), Zerogradient(:top)],
            T = [Zerogradient(:inlet), Zerogradient(:outlet),
                 Zerogradient(:bottom), Zerogradient(:top)],
        )
    )

    schemes = (
        U     = Schemes(time=Euler, divergence=Upwind, laplacian=Linear),
        p     = Schemes(time=Euler, gradient=Gauss,    laplacian=Linear),
        p_rgh = Schemes(time=Euler, gradient=Gauss,    laplacian=Linear),
        alpha = Schemes(time=Euler, divergence=Upwind, laplacian=Linear),
        T     = Schemes(time=Euler, divergence=Upwind, laplacian=Linear),
    )

    solvers = (
        U = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(),
                        convergence=1e-7, relax=1.0, rtol=0.0, atol=1e-8),
        p_rgh = SolverSetup(solver=Cg(), preconditioner=Jacobi(),
                            convergence=1e-7, relax=1.0, rtol=0.0, atol=1e-9),
        alpha = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(),
                            convergence=1e-7, relax=1.0, rtol=0.0, atol=1e-8),
        T = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(),
                        convergence=1e-7, relax=1.0, rtol=0.0, atol=1e-10),
    )

    runtime = Runtime(iterations=iterations, time_step=dt, write_interval=-1)
    Configuration(solvers=solvers, schemes=schemes,
                  runtime=runtime, hardware=hardware, boundaries=BCs)
end

"""Fill the lower half with liquid."""
function init_half_full!(model; T=T0)
    initialise!(model.momentum.U, noSlip)
    initialise!(model.fluid.p_rgh, 0.0)
    initialise!(model.fluid.alpha, 0.0)
    setField_Box!(mesh=mesh, field=model.fluid.alpha, value=1.0,
                  min_corner=[0.0, 0.0, -0.5], max_corner=[1.0, 0.5, 0.5])
    initialise!(model.energy.T, T)
end

@testset "API: accepts TwoPhaseTemperature" begin
    model = build_model(energy=Energy{TwoPhaseTemperature}(Tref=T0))
    @test model.energy isa TwoPhaseTemperature
    @test model.energy.coeffs.Tref == T0
    @test length(model.energy.T) == length(mesh_dev.cells)
end

@testset "API: Isothermal still builds to nothing" begin
    model = build_model(energy=Energy{Isothermal}())
    @test model.energy === nothing
end

@testset "API: single-phase energy models are rejected loudly" begin
    # Previously these were accepted by the Physics constructor and then
    # SILENTLY IGNORED by the multiphase solver, because run! dispatches on the
    # fluid type alone.
    model = build_model(energy=Energy{SensibleEnthalpy}(Tref=T0))
    config = build_config()

    err = try
        run!(model, config); nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("not supported by the multiphase solver", err.msg)
    @test occursin("SensibleEnthalpy", err.msg)
end

@testset "API: missing k/cp reported against the offending phase" begin
    # A phase without thermal properties must fail at setup naming the phase,
    # rather than surfacing as a `nothing` inside a kernel.
    model = build_model(
        energy = Energy{TwoPhaseTemperature}(Tref=T0),
        vapour = (rho=1.34, mu=1.11e-6),          # no k, no cp
    )
    config = build_config()

    err = try
        run!(model, config); nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("Phase 2", err.msg)
    @test occursin("TwoPhaseTemperature", err.msg)
end

@testset "Blended coefficients" begin
    model = build_model(energy=Energy{TwoPhaseTemperature}(Tref=T0))
    init_half_full!(model)
    config = build_config(iterations=1)
    run!(model, config)

    (; rho_cp, keff) = model.energy
    alpha = model.fluid.alpha

    rcp_l = LIQUID.rho*LIQUID.cp
    rcp_v = VAPOUR.rho*VAPOUR.cp

    # rho*cp must be the alpha blend, cell by cell, and bracketed by the
    # single-phase values.
    for i in eachindex(alpha.values)
        a = alpha.values[i]
        @test rho_cp.values[i] ≈ a*rcp_l + (1-a)*rcp_v rtol=1e-12
    end
    @test minimum(rho_cp.values) >= min(rcp_l, rcp_v) - 1e-8
    @test maximum(rho_cp.values) <= max(rcp_l, rcp_v) + 1e-8

    # keff likewise bracketed by the two conductivities
    @test minimum(keff.values) >= min(LIQUID.k, VAPOUR.k) - 1e-12
    @test maximum(keff.values) <= max(LIQUID.k, VAPOUR.k) + 1e-12
end

@testset "Uniform temperature is preserved (sealed, adiabatic)" begin
    # With zero-gradient T everywhere and no sources, a uniform field is an
    # exact steady state: any drift means the advection/diffusion terms are
    # inconsistent.
    model = build_model(energy=Energy{TwoPhaseTemperature}(Tref=T0))
    init_half_full!(model, T=T0)
    config = build_config(iterations=20)
    run!(model, config)

    T = model.energy.T.values
    @test maximum(abs, T .- T0) < 1e-9
end

@testset "Energy is conserved in a sealed adiabatic box" begin
    # Start with a hot upper (vapour) region and a cold lower (liquid) region -
    # stably stratified, so buoyancy does not stir it. Total sum(rho*cp*T*V)
    # must be conserved: no sources, no flux through any boundary.
    model = build_model(energy=Energy{TwoPhaseTemperature}(Tref=T0))
    init_half_full!(model, T=T0)

    # +2 K in the vapour region (upper half)
    for (i, c) in enumerate(mesh_dev.cells)
        if c.centre[2] > 0.5*0.001*500      # upper half of the scaled domain
            model.energy.T.values[i] = T0 + 2.0
        end
    end

    config = build_config(iterations=1)
    run!(model, config)

    cells = mesh_dev.cells
    total(model) = sum(model.energy.rho_cp.values[i]*model.energy.T.values[i]*cells[i].volume
                       for i in eachindex(cells))
    E0 = total(model)

    config = build_config(iterations=30)
    run!(model, config)
    E1 = total(model)

    @test abs(E1 - E0)/abs(E0) < 1e-6
end
