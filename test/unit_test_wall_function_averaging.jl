using XCALibre
using Test

# A cell at the junction of two wall-function patches owns a face on each. The value
# written for that cell must be the mean of the two face contributions rather than
# whichever face happened to write last.

@testset "Wall-function production averages over the faces a cell owns" begin
    mesh = UNV2D_mesh(joinpath(pkgdir(XCALibre, "examples/0_GRIDS"),
                               "flatplate_2D_highRe.unv"), scale=0.001)
    backend  = CPU()
    hardware = Hardware(backend=backend, workgroup=length(mesh.cells) ÷ Threads.nthreads())
    config   = (hardware = hardware,)

    nu = 1e-5
    U0 = [10.0, 0.0, 0.0]

    faces_of(name) = mesh.boundaries[boundary_index(mesh.boundaries, name)].IDs_range
    # Put both patches onto the log branch, or the production is zero and the test passes vacuously.
    (; cmu) = KWallFunction(:wall).value
    delta_min = minimum(mesh.faces[f].delta for name in (:wall, :outlet) for f in faces_of(name))
    k0 = (50*nu/(cmu^0.25*delta_min))^2

    function production(k_bcs)
        BCs = assign(region=mesh, (
            U     = [Dirichlet(:inlet, U0), Extrapolated(:outlet), Wall(:wall, [0.0, 0.0, 0.0]), Extrapolated(:top)],
            p     = [Extrapolated(:inlet), Dirichlet(:outlet, 0.0), Extrapolated(:wall), Extrapolated(:top)],
            k     = k_bcs,
            omega = [Dirichlet(:inlet, 1000.0), Extrapolated(:outlet), OmegaWallFunction(:wall), Extrapolated(:top)],
            nut   = [Extrapolated(:inlet), Extrapolated(:outlet), NutWallFunction(:wall), Extrapolated(:top)],
        ))
        model = Physics(
            time       = Steady(),
            fluid      = Fluid{Incompressible}(nu=nu),
            turbulence = RANS{KOmega}(),
            energy     = Energy{Isothermal}(),
            domain     = mesh,
        )
        initialise!(model.momentum.U, U0)
        initialise!(model.turbulence.k, k0)
        initialise!(model.turbulence.nut, k0/1000.0)
        P = ScalarField(mesh)
        XCALibre.ModelPhysics.correct_production!(P, BCs.k, model, nothing, config)
        return P.values
    end

    base = [Dirichlet(:inlet, 1.0), Extrapolated(:outlet), KWallFunction(:wall), Extrapolated(:top)]
    only_wall = production(base)
    only_out  = production([Dirichlet(:inlet, 1.0), KWallFunction(:outlet), Extrapolated(:wall), Extrapolated(:top)])
    both      = production([Dirichlet(:inlet, 1.0), KWallFunction(:outlet), KWallFunction(:wall), Extrapolated(:top)])

    cells_of(name) = Set(mesh.boundary_cellsID[f] for f in faces_of(name))
    shared = collect(intersect(cells_of(:wall), cells_of(:outlet)))

    # Non-vacuous: the patches really do share a cell, and it produces k.
    @test !isempty(shared)
    @test all(>(0), only_wall[shared])

    # Corner cells take the mean of both patches; cells on one patch only are untouched.
    @test both[shared] ≈ (only_wall[shared] .+ only_out[shared]) ./ 2 rtol=1e-12
    wall_only_cells = collect(setdiff(cells_of(:wall), cells_of(:outlet)))
    @test both[wall_only_cells] == only_wall[wall_only_cells]
end
