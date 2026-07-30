# Tests for the FixedHeatFlux boundary condition (step 4 of the LH2 tank work).
#
# The central test is an exact energy balance. With gravity switched off there is
# no buoyancy and hence no flow, so a sealed box that is adiabatic everywhere
# except one wall must gain energy at exactly
#
#     dE/dt = q * A_wall,      E = sum_i rho_cp_i * T_i * V_i
#
# That pins both the magnitude and the SIGN of the boundary coefficient, which is
# the part that is easy to get backwards (the contribution is `-term.sign*q*area`
# on the RHS; see fixedHeatFlux.jl).

using XCALibre
using Test

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "quad40.unv"), scale=0.001)

backend = CPU(); workgroup = AutoTune()
hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

noSlip = [0.0, 0.0, 0.0]
LIQUID = (rho=70.8, mu=13.2e-6, k=0.100, cp=9660.0)
T0 = 20.43

# Zero gravity: isolates the boundary condition from buoyancy-driven transport.
NO_GRAVITY = Gravity([0.0, 0.0, 0.0])

"""Patch area, summed from the mesh so the test does not hardcode the domain."""
function patch_area(m, name::Symbol)
    b = only(bb for bb in m.boundaries if bb.name === name)
    sum(m.faces[fID].area for fID in b.IDs_range)
end

"""Total thermal energy, sum(rho*cp*T*V)."""
function total_energy(model)
    cells = model.domain.cells
    rho_cp = model.energy.rho_cp.values
    T = model.energy.T.values
    sum(rho_cp[i]*T[i]*cells[i].volume for i in eachindex(cells))
end

function build(; heated_bc, iterations, dt)
    model = Physics(
        time = Transient(),
        fluid = Fluid{Multiphase}(
            model = VOF(cAlpha=1.0, sigma=0.0),
            phases = (Phase(; LIQUID...), Phase(; LIQUID...)),  # single effective phase
            gravity = NO_GRAVITY
        ),
        turbulence = RANS{Laminar}(),
        energy = Energy{TwoPhaseTemperature}(Tref=T0),
        domain = mesh_dev
    )

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
                 heated_bc,                      # :bottom
                 Zerogradient(:top)],
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
                        convergence=1e-9, relax=1.0, rtol=0.0, atol=1e-12),
        p_rgh = SolverSetup(solver=Cg(), preconditioner=Jacobi(),
                            convergence=1e-9, relax=1.0, rtol=0.0, atol=1e-12),
        alpha = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(),
                            convergence=1e-9, relax=1.0, rtol=0.0, atol=1e-12),
        T = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(),
                        convergence=1e-12, relax=1.0, rtol=0.0, atol=1e-14),
    )

    runtime = Runtime(iterations=iterations, time_step=dt, write_interval=-1)
    config = Configuration(solvers=solvers, schemes=schemes,
                           runtime=runtime, hardware=hardware, boundaries=BCs)

    initialise!(model.momentum.U, noSlip)
    initialise!(model.fluid.p_rgh, 0.0)
    initialise!(model.fluid.alpha, 1.0)      # single phase throughout
    initialise!(model.energy.T, T0)

    return model, config
end

A_WALL = patch_area(mesh_dev, :bottom)
DT = 0.01
NSTEPS = 100

@testset "FixedHeatFlux: construction" begin
    bc = FixedHeatFlux(:tankWall, 3.5)
    @test bc.ID === :tankWall
    @test bc.value == 3.5
    @test bc isa XCALibre.Discretise.AbstractNeumann
end

@testset "FixedHeatFlux: exact energy balance (heating)" begin
    q = 1000.0                       # [W/m^2] into the domain
    model, config = build(heated_bc=FixedHeatFlux(:bottom, q), iterations=NSTEPS, dt=DT)

    # rho_cp is only filled once the solver has run, so take E0 after one step
    # and measure the increment over the remaining steps.
    _, cfg1 = build(heated_bc=FixedHeatFlux(:bottom, q), iterations=1, dt=DT)
    run!(model, cfg1)
    E0 = total_energy(model)

    run!(model, config)
    E1 = total_energy(model)

    expected = q*A_WALL*NSTEPS*DT
    measured = E1 - E0

    @info "FixedHeatFlux energy balance" A_WALL expected measured ratio=measured/expected
    @test measured > 0                                   # sign: heating
    @test isapprox(measured, expected; rtol=1e-6)
end

@testset "FixedHeatFlux: exact energy balance (cooling)" begin
    # A negative flux must remove energy at the same exact rate.
    q = -500.0
    model, config = build(heated_bc=FixedHeatFlux(:bottom, q), iterations=NSTEPS, dt=DT)

    _, cfg1 = build(heated_bc=FixedHeatFlux(:bottom, q), iterations=1, dt=DT)
    run!(model, cfg1)
    E0 = total_energy(model)

    run!(model, config)
    E1 = total_energy(model)

    expected = q*A_WALL*NSTEPS*DT
    measured = E1 - E0

    @test measured < 0                                   # sign: cooling
    @test isapprox(measured, expected; rtol=1e-6)
end

@testset "FixedHeatFlux: q = 0 reduces to Zerogradient" begin
    # The q -> 0 limit is an independent check on the coefficient: it must give
    # bit-identical results to an actual Zerogradient boundary.
    model_q, config_q = build(heated_bc=FixedHeatFlux(:bottom, 0.0),
                              iterations=NSTEPS, dt=DT)
    run!(model_q, config_q)

    model_z, config_z = build(heated_bc=Zerogradient(:bottom),
                              iterations=NSTEPS, dt=DT)
    run!(model_z, config_z)

    @test model_q.energy.T.values == model_z.energy.T.values
    # and nothing should have happened at all, starting from uniform T
    @test maximum(abs, model_q.energy.T.values .- T0) < 1e-9
end

@testset "FixedHeatFlux: independent of conductivity" begin
    # Prescribing the flux prescribes the whole diffusive term, so the imposed
    # wall heat load must not depend on k. Only the resulting temperature
    # DISTRIBUTION should change.
    q = 1000.0
    expected = q*A_WALL*NSTEPS*DT

    for kval in (0.05, 0.100, 1.0)
        liquid = (rho=LIQUID.rho, mu=LIQUID.mu, k=kval, cp=LIQUID.cp)

        model = Physics(
            time = Transient(),
            fluid = Fluid{Multiphase}(
                model = VOF(cAlpha=1.0, sigma=0.0),
                phases = (Phase(; liquid...), Phase(; liquid...)),
                gravity = NO_GRAVITY
            ),
            turbulence = RANS{Laminar}(),
            energy = Energy{TwoPhaseTemperature}(Tref=T0),
            domain = mesh_dev
        )

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
                     FixedHeatFlux(:bottom, q), Zerogradient(:top)],
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
                            convergence=1e-9, relax=1.0, rtol=0.0, atol=1e-12),
            p_rgh = SolverSetup(solver=Cg(), preconditioner=Jacobi(),
                                convergence=1e-9, relax=1.0, rtol=0.0, atol=1e-12),
            alpha = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(),
                                convergence=1e-9, relax=1.0, rtol=0.0, atol=1e-12),
            T = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(),
                            convergence=1e-12, relax=1.0, rtol=0.0, atol=1e-14),
        )

        initialise!(model.momentum.U, noSlip)
        initialise!(model.fluid.p_rgh, 0.0)
        initialise!(model.fluid.alpha, 1.0)
        initialise!(model.energy.T, T0)

        cfg1 = Configuration(solvers=solvers, schemes=schemes,
                             runtime=Runtime(iterations=1, time_step=DT, write_interval=-1),
                             hardware=hardware, boundaries=BCs)
        run!(model, cfg1)
        E0 = total_energy(model)

        cfgN = Configuration(solvers=solvers, schemes=schemes,
                             runtime=Runtime(iterations=NSTEPS, time_step=DT, write_interval=-1),
                             hardware=hardware, boundaries=BCs)
        run!(model, cfgN)

        @test isapprox(total_energy(model) - E0, expected; rtol=1e-6)
    end
end
