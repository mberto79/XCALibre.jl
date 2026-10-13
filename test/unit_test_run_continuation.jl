# A run! continued on the same model must retrace one uninterrupted run: same fields and the same
# residuals in the second chunk. Re-initialising the fields between runs must start afresh.

# Inviscid Mach 2 flow over a 15° compression corner (steady compressible)
function continuation_corner(iterations)
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
        U = solver(0.7, nothing), p = solver(0.3, (0.5*p_inf, 5*p_inf)),
        he = solver(0.3, (800.0, 3000.0)))
    schemes = (
        U = Schemes(divergence=BoundedUpwind), p = Schemes(divergence=Upwind),
        he = Schemes(divergence=BoundedUpwind))
    config = Configuration(
        solvers=solvers, schemes=schemes, boundaries=boundaries,
        runtime=Runtime(iterations=iterations, write_interval=-1, time_step=1),
        hardware=Hardware(backend=CPU(), workgroup=workgroupsize(mesh)))

    init!(model) = begin
        initialise!(model.momentum.U, [U_inf, 0.0, 0.0])
        initialise!(model.momentum.p, p_inf)
        initialise!(model.energy.T, T_inf)
    end
    init!(model)
    model, config, init!
end

# Laminar backward-facing step (steady incompressible)
function continuation_bfs(iterations)
    mesh = UNV2D_mesh(pkgdir(XCALibre, "examples/0_GRIDS/backwardFacingStep_10mm.unv"), scale=0.001)
    velocity = [0.5, 0.0, 0.0]

    model = Physics(
        time = Steady(),
        fluid = Fluid{Incompressible}(nu=1e-3),
        turbulence = RANS{Laminar}(),
        energy = Energy{Isothermal}(),
        domain = mesh
        )

    boundaries = assign(region=mesh, (
        U = [Dirichlet(:inlet, velocity), Extrapolated(:outlet), Wall(:wall, [0.0, 0.0, 0.0]),
            Symmetry(:top)],
        p = [Extrapolated(:inlet), Dirichlet(:outlet, 0.0), Extrapolated(:wall), Symmetry(:top)]
        ))

    solvers = (
        U = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-10,
            relax=0.8, rtol=1e-1),
        p = SolverSetup(solver=Cg(), preconditioner=Jacobi(), convergence=1e-10,
            relax=0.2, rtol=1e-2)
        )
    config = Configuration(
        solvers=solvers, schemes=(U=Schemes(divergence=Linear), p=Schemes()), boundaries=boundaries,
        runtime=Runtime(iterations=iterations, write_interval=-1, time_step=1),
        hardware=Hardware(backend=CPU(), workgroup=workgroupsize(mesh)))

    init!(model) = begin
        initialise!(model.momentum.U, velocity)
        initialise!(model.momentum.p, 0.0)
    end
    init!(model)
    model, config, init!
end

snapshot(model) = copy.((model.momentum.U.x.values, model.momentum.U.y.values,
    model.momentum.p.values, hasproperty(model.energy, :T) ? model.energy.T.values : Float64[]))

# Threaded boundary accumulation is atomic, so its summation order can vary between runs (~1e-14
# in a field); a continued run that rebuilds its face flux from U differs by ~1e-5 here.
reldiff(a, b) = maximum(abs.(a .- b); init=0.0)/max(maximum(abs.(a); init=0.0), eps())
fields_match(m1, m2) = all(reldiff.(snapshot(m1), snapshot(m2)) .<= 1e-10)
residuals_match(r1, r2) = all(reldiff(r1[k], r2[k]) <= 1e-8 for k ∈ (:Ux, :Uy, :p))

function continuation_check(case, n)
    model1, config1, _ = case(2n)
    r1 = run!(model1, config1; progress=false)

    model2, config2, init! = case(n)
    run!(model2, config2; progress=false)
    r2 = run!(model2, config2; progress=false)
    fields = fields_match(model1, model2)
    residuals = residuals_match(map(r -> r[n+1:2n], r1), r2)

    # a run after re-initialising the fields must equal a fresh run
    model3, config3, _ = case(n)
    r3 = run!(model3, config3; progress=false)
    init!(model2)
    r4 = run!(model2, config2; progress=false)
    reinitialised = fields_match(model3, model2) && residuals_match(r3, r4)

    (; fields, residuals, reinitialised)
end

@testset "Compressible steady (compression corner)" begin
    res = continuation_check(continuation_corner, 300)
    @test res.fields
    @test res.residuals
    @test res.reinitialised
end

@testset "Incompressible steady (backward-facing step)" begin
    res = continuation_check(continuation_bfs, 100)
    @test res.fields
    @test res.residuals
    @test res.reinitialised
end
