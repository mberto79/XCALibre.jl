using XCALibre
using LinearAlgebra
using StaticArrays
using Random
using Test

# Boundary contributions are summed per owner cell in face order (no atomics), so they equal a
# serial face loop for any workgroup size and thread count; atomic face loops made corner cells
# depend on the thread schedule, and unconverged iterations amplified that round-off

config_for(workgroup) = (hardware = Hardware(backend=CPU(), workgroup=workgroup),)
grids = pkgdir(XCALibre, "examples/0_GRIDS")

@testset "Boundary faces grouped by owner cell" begin
    mesh = UNV2D_mesh(joinpath(grids, "backwardFacingStep_10mm.unv"), scale=0.001)
    (; boundary_cellsID, boundary_cells, boundary_cell_faces_range, boundary_cell_faces) = mesh
    nb = length(boundary_cellsID)
    @test sort(boundary_cell_faces) == 1:nb                     # every boundary face once
    @test issorted(boundary_cells) && allunique(boundary_cells)
    @test Set(boundary_cells) == Set(boundary_cellsID)
    @test first(first(boundary_cell_faces_range)) == 1 && last(last(boundary_cell_faces_range)) == nb
    @test all(zip(boundary_cells, boundary_cell_faces_range)) do (c, r)
        fs = boundary_cell_faces[r]
        issorted(fs) && all(boundary_cellsID[f] == c for f ∈ fs)
    end
    @test any(length(r) > 1 for r ∈ boundary_cell_faces_range)  # corner cells exist (non-vacuous)

    # reordering renumbers the cells and the boundary faces: the grouping follows
    reorder_mesh!(mesh)
    @test (mesh.boundary_cells, mesh.boundary_cell_faces_range, mesh.boundary_cell_faces) ==
        XCALibre.Mesh.boundary_cell_faces(mesh.boundary_cellsID)
end

@testset "Boundary accumulation equals a serial face loop ($grid)" for grid ∈ (
        "backwardFacingStep_10mm.unv", "trig40.unv")
    mesh = UNV2D_mesh(joinpath(grids, grid), scale=0.001)
    (; cells, faces, cell_faces, cell_nsign, boundary_cellsID) = mesh
    ncells, nfaces, nb = length(cells), length(faces), length(boundary_cellsID)
    workgroups = (1, 3, 16, AutoTune())
    repeats = 20
    rng = Xoshiro(7)

    phif = FaceScalarField(mesh); phif.values .= randn(rng, nfaces)
    psif = FaceVectorField(mesh)
    psif.x.values .= randn(rng, nfaces); psif.y.values .= randn(rng, nfaces)

    # serial reference with the kernels' arithmetic: internal faces per cell, then each boundary
    # face added to its owner in face order
    gx, gy, dv = zeros(ncells), zeros(ncells), zeros(ncells)
    for i ∈ 1:ncells
        g = zero(SVector{3,Float64})
        s = 0.0
        for k ∈ cells.faces_range[i]
            f = cell_faces[k]
            g += phif.values[f]*(faces.area[f]*faces.normal[f]*cell_nsign[k])
            s += psif[f]⋅(faces.area[f]*faces.normal[f])*cell_nsign[k]
        end
        g /= cells.volume[i]
        gx[i], gy[i], dv[i] = g[1], g[2], s/cells.volume[i]
    end
    for f ∈ 1:nb
        c = boundary_cellsID[f]
        res = phif.values[f]*(faces.area[f]*faces.normal[f])/cells.volume[c]
        gx[c] += res[1]; gy[c] += res[2]
        dv[c] += psif[f]⋅(faces.area[f]*faces.normal[f])/cells.volume[c]
    end

    grad = Grad{Gauss}(ScalarField(mesh))
    divergence = ScalarField(mesh)
    @test all(1:repeats) do _
        all(workgroups) do workgroup
            config = config_for(workgroup)
            grad!(grad, phif, config)
            div!(divergence, psif, config)
            grad.result.x.values == gx && grad.result.y.values == gy && divergence.values == dv
        end
    end

    # least squares: same result for every schedule
    phi = ScalarField(mesh); phi.values .= randn(rng, ncells)
    lsq = Grad{LeastSquares}(phi)
    grad!(lsq, phif, phi, 0.0, config_for(ncells))
    lx = copy(lsq.result.x.values)
    @test all(1:repeats) do _
        all(workgroups) do workgroup
            grad!(lsq, phif, phi, 0.0, config_for(workgroup))
            lsq.result.x.values == lx
        end
    end
end

# A whole steady run is bitwise the same for any workgroup size, so also from run to run and
# for any thread count at which the Krylov reductions are not split (small meshes).
function steady_bfs(workgroup, iterations)
    mesh = UNV2D_mesh(joinpath(grids, "backwardFacingStep_10mm.unv"), scale=0.001)
    velocity = [0.5, 0.0, 0.0]
    model = Physics(
        time = Steady(), fluid = Fluid{Incompressible}(nu=1e-3), turbulence = RANS{Laminar}(),
        energy = Energy{Isothermal}(), domain = mesh)
    boundaries = assign(region=mesh, (
        U = [Dirichlet(:inlet, velocity), Extrapolated(:outlet), Wall(:wall, [0.0, 0.0, 0.0]),
            Symmetry(:top)],
        p = [Extrapolated(:inlet), Dirichlet(:outlet, 0.0), Extrapolated(:wall), Symmetry(:top)]))
    solvers = (
        U = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-10,
            relax=0.8, rtol=1e-1),
        p = SolverSetup(solver=Cg(), preconditioner=Jacobi(), convergence=1e-10,
            relax=0.2, rtol=1e-2))
    config = Configuration(
        solvers=solvers, schemes=(U=Schemes(divergence=Linear), p=Schemes(gradient=LeastSquares)),
        boundaries=boundaries,
        runtime=Runtime(iterations=iterations, write_interval=-1, time_step=1),
        hardware=Hardware(backend=CPU(), workgroup=workgroup))
    initialise!(model.momentum.U, velocity)
    initialise!(model.momentum.p, 0.0)
    residuals = run!(model, config; progress=false, ncorrectors=1)
    (copy(model.momentum.U.x.values), copy(model.momentum.U.y.values), copy(model.momentum.p.values)),
        residuals
end

@testset "Steady run independent of the schedule" begin
    fields, residuals = steady_bfs(AutoTune(), 80)
    for workgroup ∈ (AutoTune(), 7, 64)
        f, r = steady_bfs(workgroup, 80)
        @test f == fields
        @test all(r[k] == residuals[k] for k ∈ (:Ux, :Uy, :p))
    end
end
