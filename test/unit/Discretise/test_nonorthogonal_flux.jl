# With non-orthogonal correctors the converged face mass flux must balance to round-off, not to
# rtol times the non-orthogonal source (an uncorrected first pressure solve leaves that error).
# Inviscid Mach 2 flow over a 15° compression corner (non-orthogonal cells along the ramp).
function nonorthogonal_corner_continuity(; ncorrectors=1, iterations=1000)
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

    # loose relative tolerance on p: the continuity error of the old scheme scales with it
    solver(relax, limit, rtol, atol) = SolverSetup(
        solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-12, relax=relax,
        limit=limit, rtol=rtol, atol=atol)
    solvers = (
        U = solver(0.7, nothing, 0.0, 1e-2), p = solver(0.3, (0.5*p_inf, 5*p_inf), 1e-2, 1e-20),
        he = solver(0.3, (800.0, 3000.0), 0.0, 1e-2))
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
    run!(model, config; ncorrectors=ncorrectors, progress=false)

    return max_continuity_error(mesh, model.momentum.mdotf.values)
end

# Incompressible laminar flow over the same corner with `simple!` or `piso!`
function nonorthogonal_corner_continuity_incompressible(;
    steady=true, ncorrectors=1, iterations=500)
    mesh = UNV2D_mesh(
        pkgdir(XCALibre, "examples/0_GRIDS/compression_corner_2d_32_64_SF4.unv"), scale=0.001)
    U_inf = 1.0
    model = Physics(
        time = steady ? Steady() : Transient(),
        fluid = Fluid{Incompressible}(nu=1e-4),
        turbulence = RANS{Laminar}(),
        energy = Energy{Isothermal}(),
        domain = mesh
        )
    boundaries = assign(region=mesh, (
        U = [Dirichlet(:inlet, [U_inf, 0.0, 0.0]), Zerogradient(:outlet), Zerogradient(:top),
            Slip(:wall)],
        p = [Zerogradient(:inlet), Dirichlet(:outlet, 0.0), Zerogradient(:top),
            Zerogradient(:wall)]
        ))
    solvers = (
        U = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-12,
            relax=steady ? 0.7 : 1.0, rtol=0.0, atol=1e-10),
        p = SolverSetup(solver=Cg(), preconditioner=Jacobi(), convergence=1e-12,
            relax=steady ? 0.3 : 1.0, rtol=1e-2, atol=1e-20))
    time_scheme = steady ? SteadyState : Euler
    schemes = (
        U = Schemes(divergence=Upwind, time=time_scheme), p = Schemes(time=time_scheme))
    config = Configuration(
        solvers=solvers, schemes=schemes, boundaries=boundaries,
        runtime=Runtime(
            iterations=iterations, write_interval=-1, time_step=steady ? 1 : 1e-3),
        hardware=Hardware(backend=CPU(), workgroup=1024))

    initialise!(model.momentum.U, [U_inf, 0.0, 0.0])
    initialise!(model.momentum.p, 0.0)
    run!(model, config; ncorrectors=ncorrectors, progress=false)
    return max_continuity_error(mesh, model.momentum.mdotf.values)
end

# per-cell continuity error of a face mass flux, relative to the cell outflow
function max_continuity_error(mesh, mdot)
    (; cells, cell_faces, cell_nsign, boundary_cellsID) = mesh
    imbalance = zeros(length(cells)); outflow = zeros(length(cells))
    for cID ∈ eachindex(cells), fi ∈ cells[cID].faces_range
        m = cell_nsign[fi]*mdot[cell_faces[fi]]
        imbalance[cID] += m; outflow[cID] += max(m, 0.0)
    end
    for fID ∈ eachindex(boundary_cellsID)
        cID = boundary_cellsID[fID]
        imbalance[cID] += mdot[fID]; outflow[cID] += max(mdot[fID], 0.0)
    end
    return maximum(abs.(imbalance)./outflow)
end

# ncorrectors=2 runs the first corrector too, so it also catches the single-corrector error.
# Before the fix: compressible 2.9e-6, simple! 4.4e-6, piso! (50 steps) 3.1e-6; now ~1e-9 or below
@test nonorthogonal_corner_continuity(ncorrectors=2, iterations=400) < 1e-8
@test nonorthogonal_corner_continuity_incompressible(ncorrectors=2, iterations=100) < 1e-8
@test nonorthogonal_corner_continuity_incompressible(
    steady=false, ncorrectors=1, iterations=30) < 1e-8
