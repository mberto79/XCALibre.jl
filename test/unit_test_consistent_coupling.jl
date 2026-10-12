# Uniform Mach 0.84 stream through an empty box, an exact steady solution of the compressible
# equations. With relaxation (0.6, 0.3) the default coupling amplifies round-off from iteration to
# iteration; the consistent coupling must hold the free stream to round-off.
function uniform_stream_box(consistent; iterations=60)
    mesh = UNV3D_mesh(pkgdir(XCALibre, "examples/0_GRIDS/3d_box_1000x1000x1000mm_5.unv"), scale=0.001)
    gamma, cp, Pr = 1.4, 1005.0, 0.72
    R = cp*(1 - 1/gamma)
    T_inf, p_inf = 255.56, 80600.0
    U_inf = 0.84*sqrt(gamma*R*T_inf)
    velocity = [U_inf, 0.0, 0.0]

    model = Physics(
        time = Steady(),
        fluid = Fluid{Compressible}(nu=1.5e-5, cp=cp, gamma=gamma, Pr=Pr),
        turbulence = RANS{Laminar}(),
        energy = Energy{SensibleEnthalpy}(Tref=0.0),
        domain = mesh
        )

    enthalpy = Enthalpy(cp=cp, Tref=0.0)
    sides = (:y_min, :y_max, :z_min, :z_max)
    boundaries = assign(region=mesh, (
        U = [Dirichlet(:x_min, velocity), Zerogradient(:x_max), [Symmetry(s) for s ∈ sides]...],
        p = [Zerogradient(:x_min), Dirichlet(:x_max, p_inf), [Symmetry(s) for s ∈ sides]...],
        he = [
            FixedTemperature(:x_min, T=T_inf, enthalpy), Zerogradient(:x_max),
            [Symmetry(s) for s ∈ sides]...]
        ))

    solver(relax, rtol) = SolverSetup(
        solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-12, relax=relax,
        rtol=rtol, atol=1e-14)
    solvers = (U=solver(0.6, 1e-1), p=solver(0.3, 1e-6), he=solver(0.3, 1e-1))
    upwind = Schemes(divergence=Upwind, gradient=Gauss)
    config = Configuration(
        solvers=solvers, schemes=(U=upwind, p=upwind, he=upwind), boundaries=boundaries,
        runtime=Runtime(iterations=iterations, write_interval=-1, time_step=1),
        hardware=Hardware(backend=CPU(), workgroup=workgroupsize(mesh)))

    initialise!(model.momentum.U, velocity)
    initialise!(model.momentum.p, p_inf)
    initialise!(model.energy.T, T_inf)

    run!(model, config; progress=false, consistent=consistent)

    (; U, p) = model.momentum
    dU = maximum(abs.(U.x.values .- U_inf))/U_inf
    dp = maximum(abs.(p.values .- p_inf))/p_inf
    return dU, dp
end

dU, dp = uniform_stream_box(true)
@test dU < 1e-8
@test dp < 1e-8

dU, dp = uniform_stream_box(false) # round-off grows by about 1.5 per iteration
@test max(dU, dp) > 1e-6

import Accessors

# Incompressible laminar backward-facing step: the consistent coupling converges without pressure
# under-relaxation to the solution of the default coupling (relaxation 0.8, 0.2).
include(joinpath(@__DIR__, "distributed", "psimple_case.jl"))

function bfs_simple(consistent, p_relax, iterations)
    model, config = incompressible_case(bfs_mesh(), bfs_bcs; iterations=iterations)
    p_solver = Accessors.setproperties(config.solvers.p, relax=p_relax)
    config = Accessors.@set config.solvers = (U=config.solvers.U, p=p_solver)
    simple!(model, config; consistent=consistent, progress=false)
    return Array(model.momentum.U.x.values), Array(model.momentum.p.values)
end

Ux_ref, p_ref = bfs_simple(false, 0.2, 1500)
Ux, p = bfs_simple(true, 1.0, 800)
@test maximum(abs.(Ux .- Ux_ref)) < 1e-5
@test maximum(abs.(p .- p_ref)) < 1e-5
