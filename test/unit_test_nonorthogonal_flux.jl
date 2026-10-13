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
        hardware=Hardware(backend=CPU(), workgroup=workgroupsize(mesh)))

    initialise!(model.momentum.U, [U_inf, 0.0, 0.0])
    initialise!(model.momentum.p, p_inf)
    initialise!(model.energy.T, T_inf)
    run!(model, config; ncorrectors=ncorrectors, progress=false)

    # per-cell continuity error of the final flux, relative to the cell outflow
    mdot = model.momentum.mdotf.values
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

# Before the fix: 1.8e-4 (ncorrectors=1) and 2.9e-6 (ncorrectors=2); now round-off (~2e-12)
@test nonorthogonal_corner_continuity(ncorrectors=1) < 1e-8
@test nonorthogonal_corner_continuity(ncorrectors=2, iterations=600) < 1e-8
