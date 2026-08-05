using XCALibre
using Test
using LinearAlgebra

# =============================================================================
#  LH2 vertical heated pipe - 90 degree O-grid sector mesh
# =============================================================================
#
#  Geometric validation of the mesh produced by
#      examples/0_GRIDS/lh2_pipe_sector/make_lh2_pipe_sector.jl + blockMesh
#
#  Run after building the mesh:
#      cd examples/0_GRIDS/lh2_pipe_sector
#      julia make_lh2_pipe_sector.jl && ./run_blockMesh.sh
#      cd ../../.. && julia --project=. test/unit_test_lh2_pipe_sector_mesh.jl
#
#  The mesh is not in version control (blockMesh output is a build product), so
#  the whole file skips cleanly when it is absent rather than failing.
# =============================================================================

const PIPE_GRID_DIR = joinpath(pkgdir(XCALibre, "examples/0_GRIDS"), "lh2_pipe_sector")
const PIPE_POLYMESH = joinpath(PIPE_GRID_DIR, "constant", "polyMesh")

# Must match the CASE selected in the generator.
const PIPE_D = 6.0e-3
const PIPE_L_HEATED = 250.0e-3
const PIPE_L_DEV = 10.0*PIPE_D
const PIPE_L_TOTAL = PIPE_L_DEV + PIPE_L_HEATED
const PIPE_R = PIPE_D/2

# Flow conditions the generator sized the near-wall cell for.
const PIPE_U_BULK = 5.33
const PIPE_RHO_L = 56.75
const PIPE_MU_L = 6.99e-6

if !isdir(PIPE_POLYMESH)
    @info """Skipping LH2 pipe sector mesh tests: no mesh at
        $PIPE_POLYMESH

    Build it with:
        cd examples/0_GRIDS/lh2_pipe_sector
        julia make_lh2_pipe_sector.jl && ./run_blockMesh.sh"""
else

mesh = FOAM3D_mesh(PIPE_POLYMESH, scale=1.0, integer_type=Int64, float_type=Float64)

@testset "LH2 pipe sector: geometry" begin
    # A quarter of a cylinder.
    V_expected = 0.25*pi*PIPE_R^2*PIPE_L_TOTAL
    V_mesh = sum(c.volume for c in mesh.cells)
    @info "volume" V_mesh V_expected rel_error=abs(V_mesh - V_expected)/V_expected

    # An O-grid resolves a circular boundary with straight-edged cells, so the
    # volume is slightly under the analytical value; the deficit shrinks with
    # azimuthal refinement. 1% is a generous bound for the default resolution.
    @test isapprox(V_mesh, V_expected, rtol=0.01)

    # Every cell must have positive volume.
    @test all(c -> c.volume > 0, mesh.cells)

    # The sector occupies the first quadrant, from the axis out to the wall.
    xs = [c.centre[1] for c in mesh.cells]
    ys = [c.centre[2] for c in mesh.cells]
    zs = [c.centre[3] for c in mesh.cells]

    @test minimum(xs) >= -1e-12
    @test minimum(ys) >= -1e-12
    @test maximum(sqrt.(xs.^2 .+ ys.^2)) <= PIPE_R + 1e-9
    @test minimum(zs) > 0
    @test maximum(zs) < PIPE_L_TOTAL
end

@testset "LH2 pipe sector: no axis degeneracy" begin
    # The point of choosing an O-grid over a wedge. A wedge collapses onto the
    # axis, giving zero-area faces and a rank-deficient reconstruction; an
    # O-grid has no cell edge on the centreline at all, so every face metric
    # must be strictly finite and non-degenerate.
    @test all(f -> isfinite(f.area) && f.area > 0, mesh.faces)
    @test all(f -> isfinite(f.delta) && f.delta > 0, mesh.faces)
    @test all(f -> all(isfinite, f.normal), mesh.faces)
    @test all(f -> isapprox(norm(f.normal), 1.0, atol=1e-8), mesh.faces)

    # No cell centre sits exactly on the axis either.
    r_min = minimum(sqrt(c.centre[1]^2 + c.centre[2]^2) for c in mesh.cells)
    @info "closest cell centre to the axis" r_min
    @test r_min > 0
end

@testset "LH2 pipe sector: boundary patches" begin
    names = Set(b.name for b in get_boundaries(mesh.boundaries))
    for expected in (:inlet, :outlet, :pipeWall, :wallUnheated, :symmetryX, :symmetryY)
        @test expected in names
    end

    boundaries = get_boundaries(mesh.boundaries)
    patch_area(name) = begin
        idx = boundary_index(boundaries, name)
        sum(mesh.faces[fID].area for fID in boundaries[idx].IDs_range)
    end

    # Inlet and outlet are quarter discs.
    A_disc = 0.25*pi*PIPE_R^2
    @test isapprox(patch_area(:inlet), A_disc, rtol=0.01)
    @test isapprox(patch_area(:outlet), A_disc, rtol=0.01)

    # The HEATED wall area sets the total power for a given flux, so it has to
    # be right: a quarter of the lateral surface over the heated length only.
    A_heated = 0.25*2pi*PIPE_R*PIPE_L_HEATED
    @info "heated wall area" patch_area(:pipeWall) A_heated
    @test isapprox(patch_area(:pipeWall), A_heated, rtol=0.01)

    A_unheated = 0.25*2pi*PIPE_R*PIPE_L_DEV
    @test isapprox(patch_area(:wallUnheated), A_unheated, rtol=0.01)

    # Symmetry planes are rectangles PIPE_R x PIPE_L_TOTAL.
    A_sym = PIPE_R*PIPE_L_TOTAL
    @test isapprox(patch_area(:symmetryX), A_sym, rtol=0.01)
    @test isapprox(patch_area(:symmetryY), A_sym, rtol=0.01)
end

@testset "LH2 pipe sector: wall functions see y+ in the 30-50 band" begin
    # The mesh exists to support a HIGH y+ wall treatment: the RPI model takes
    # its convective flux from the log branch of the thermal wall function,
    # which is only valid once the first cell centre is clear of the buffer
    # layer. Resolving finer than y+ ~ 30 would silently use the wrong formula.
    nu = PIPE_MU_L/PIPE_RHO_L
    Re = PIPE_RHO_L*PIPE_U_BULK*PIPE_D/PIPE_MU_L
    f = (0.790*log(Re) - 1.64)^-2
    u_tau = sqrt((f/8)*PIPE_U_BULK^2)

    boundaries = get_boundaries(mesh.boundaries)
    idx = boundary_index(boundaries, :pipeWall)

    # `delta` on a boundary face is the wall-normal distance to the owner cell
    # centre, which is exactly what the wall function is evaluated at.
    yplus = [mesh.faces[fID].delta*u_tau/nu for fID in boundaries[idx].IDs_range]

    @info "y+ on the heated wall" minimum(yplus) maximum(yplus) mean=sum(yplus)/length(yplus)

    @test minimum(yplus) > 30.0
    @test maximum(yplus) < 50.0
end

@testset "LH2 pipe sector: reconstruct! recovers a uniform field" begin
    # `reconstruct!` inverts a per-cell 3x3 face-normal moment matrix. It is the
    # routine that went rank-deficient on the K-Site wedge, so it is worth
    # confirming the O-grid conditions it properly: for a uniform cell vector
    # `g`, the exact face data is psif_f = area_f (g . n_f), and the
    # reconstruction must return `g` in every cell.
    backend = CPU()
    hardware = Hardware(backend=backend, workgroup=AutoTune())
    mesh_dev = adapt(backend, mesh)
    config = (hardware=hardware,)

    ws   = XCALibre.Solvers.ReconstructWorkspace(mesh_dev, backend)
    phi  = VectorField(mesh_dev)
    psif = FaceScalarField(mesh_dev)

    g = (0.0, 0.0, -9.81)
    for (i, f) in enumerate(mesh_dev.faces)
        psif.values[i] = f.area*(g[1]*f.normal[1] + g[2]*f.normal[2] + g[3]*f.normal[3])
    end

    XCALibre.Solvers.reconstruct!(phi, psif, config, ws)

    err = max(maximum(abs, phi.x.values .- g[1]),
              maximum(abs, phi.y.values .- g[2]),
              maximum(abs, phi.z.values .- g[3]))
    @info "reconstruct! uniform-field error" err
    @test err < 1e-9

    # No cell may come back identically zero - the signature of a singular
    # moment matrix silently returning nothing.
    n_zero = count(i -> phi.x.values[i] == 0 && phi.y.values[i] == 0 &&
                        phi.z.values[i] == 0, 1:length(mesh_dev.cells))
    @test n_zero == 0
end

end # isdir(PIPE_POLYMESH)
