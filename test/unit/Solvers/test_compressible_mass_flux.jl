# The compressible face mass flux must be the flux the pressure equation balances. Inviscid Mach 2
# flow over a 15° compression corner with adiabatic slip walls: the total temperature is uniform,
# and the oblique shock gives T2/T1 = 1.270 and p2/p1 = 2.196.
function compression_corner_conservation(iterations=1000)
    mesh = UNV2D_mesh(
        pkgdir(XCALibre, "examples/0_GRIDS/compression_corner_2d_32_64_SF4.unv"), scale=0.001)
    gamma, cp = 1.4, 1005.0
    R = cp*(1 - 1/gamma)
    T_inf, p_inf = 1000.0, 100000.0
    U_inf = 2*sqrt(gamma*R*T_inf)

    model = Physics(
        time = Steady(),
        fluid = Fluid{Compressible}(nu=0.0, cp=cp, gamma=gamma, Pr=0.7),
        turbulence = RANS{Laminar}(),
        energy = Energy{SensibleEnthalpy}(Tref=0.0),
        domain = mesh
        )

    boundaries = assign(region=mesh, (
        U = [Dirichlet(:inlet, [U_inf, 0.0, 0.0]), Zerogradient(:outlet), Zerogradient(:top),
            Slip(:wall)],
        p = [Dirichlet(:inlet, p_inf), Zerogradient(:outlet), Zerogradient(:top),
            Zerogradient(:wall)],
        he = [FixedTemperature(:inlet, T=T_inf, Enthalpy(cp=cp, Tref=0.0)), Zerogradient(:outlet),
            Zerogradient(:top), Zerogradient(:wall)]
        ))

    solver(relax, limit) = SolverSetup(
        solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-10, relax=relax,
        limit=limit, rtol=0.0, atol=1e-2)
    solvers = (
        U = solver(0.7, nothing), p = solver(0.3, (0.5*p_inf, 5*p_inf)), he = solver(0.3, (800.0, 3000.0)))
    schemes = (
        U = Schemes(divergence=BoundedUpwind), p = Schemes(divergence=Upwind),
        he = Schemes(divergence=BoundedUpwind))
    config = Configuration(
        solvers=solvers, schemes=schemes, boundaries=boundaries,
        runtime=Runtime(iterations=iterations, write_interval=-1, time_step=1),
        hardware=Hardware(backend=CPU(), workgroup=1024))

    initialise!(model.momentum.U, [U_inf, 0.0, 0.0])
    initialise!(model.momentum.p, p_inf)
    initialise!(model.energy.T, T_inf)
    run!(model, config; progress=false)

    (; U, p) = model.momentum
    T = model.energy.T.values
    T0_inf = T_inf + U_inf^2/(2cp)
    T0 = @. T + (U.x.values^2 + U.y.values^2)/(2cp)
    post = T .> 1150.0 # cells behind the shock
    return maximum(abs.(T0 .- T0_inf))/T0_inf, median(T[post])/T_inf, median(p.values[post])/p_inf
end

dT0, T_ratio, p_ratio = compression_corner_conservation()
@test dT0 < 1e-6
@test isapprox(T_ratio, 1.270, rtol=5e-3)
@test isapprox(p_ratio, 2.196, rtol=5e-3)
