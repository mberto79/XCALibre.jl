using XCALibre
using Test

# Metis partitions cells, not patches, so an MPI rank can hold a wall-function patch with no
# faces. Wall functions launch one kernel per patch, so an empty range must skip the launch.

@testset "Wall functions skip a patch with no faces" begin
    mesh = UNV2D_mesh(joinpath(pkgdir(XCALibre, "examples/0_GRIDS"),
                               "flatplate_2D_highRe.unv"), scale=0.001)
    backend  = CPU()
    hardware = Hardware(backend=backend, workgroup=length(mesh.cells) ÷ Threads.nthreads())
    config   = (hardware = hardware,)

    nu = 1e-5
    U0 = [10.0, 0.0, 0.0]

    faces_of(name) = mesh.boundaries[boundary_index(mesh.boundaries, name)].IDs_range
    (; cmu) = KWallFunction(:wall).value
    delta_min = minimum(mesh.faces[f].delta for name in (:wall, :outlet) for f in faces_of(name))
    k0 = (50*nu/(cmu^0.25*delta_min))^2

    empty_patch(bc) = typeof(bc)(bc.ID, bc.value, 1:0)

    function model_and_bcs(k_bcs, nut_bcs)
        BCs = assign(region=mesh, (
            U     = [Dirichlet(:inlet, U0), Extrapolated(:outlet), Wall(:wall, [0.0, 0.0, 0.0]), Extrapolated(:top)],
            p     = [Extrapolated(:inlet), Dirichlet(:outlet, 0.0), Extrapolated(:wall), Extrapolated(:top)],
            k     = k_bcs,
            omega = [Dirichlet(:inlet, 1000.0), Extrapolated(:outlet), OmegaWallFunction(:wall), Extrapolated(:top)],
            nut   = nut_bcs,
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
        return model, BCs
    end

    nut_base = [Extrapolated(:inlet), Extrapolated(:outlet), NutWallFunction(:wall), Extrapolated(:top)]

    function production(k_bcs; empty_outlet=false)
        model, BCs = model_and_bcs(k_bcs, nut_base)
        kBCs = empty_outlet ? map(bc -> bc.ID == boundary_index(mesh.boundaries, :outlet) ? empty_patch(bc) : bc, BCs.k) : BCs.k
        P = ScalarField(mesh)
        XCALibre.ModelPhysics.correct_production!(P, kBCs, model, nothing, config)
        return P.values
    end

    wall_only = [Dirichlet(:inlet, 1.0), Extrapolated(:outlet), KWallFunction(:wall), Extrapolated(:top)]
    both      = [Dirichlet(:inlet, 1.0), KWallFunction(:outlet), KWallFunction(:wall), Extrapolated(:top)]

    reference = production(wall_only)
    # Non-vacuous: the patch that is emptied would otherwise contribute.
    @test reference != production(both)
    @test production(both; empty_outlet=true) ≈ reference rtol=1e-12

    # nut: emptying the only wall-function patch must leave the face field untouched
    model, BCs = model_and_bcs(wall_only, nut_base)
    nutf = FaceScalarField(mesh)
    nut_emptied = map(bc -> bc.ID == boundary_index(mesh.boundaries, :wall) ? empty_patch(bc) : bc, BCs.nut)
    XCALibre.ModelPhysics.correct_eddy_viscosity!(nutf, nut_emptied, model, config)
    @test all(iszero, nutf.values)
    XCALibre.ModelPhysics.correct_eddy_viscosity!(nutf, BCs.nut, model, config)
    @test any(>(0), nutf.values)
end
