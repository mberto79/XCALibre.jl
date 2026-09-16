using XCALibre
using Test

# `set_production!` overwrites Pk in KWallFunction wall cells, so like the rest of the conservative
# k equation it must carry rho. Every CI case with a k wall function has rho = 1, so this calls the
# kernel directly at two densities and checks the wall-cell production scales exactly with rho.

@testset "KWallFunction production is density-weighted" begin
    mesh = UNV2D_mesh(joinpath(pkgdir(XCALibre, "examples/0_GRIDS"),
                               "flatplate_2D_highRe.unv"), scale=0.001)
    backend  = CPU()
    hardware = Hardware(backend=backend, workgroup=length(mesh.cells) ÷ Threads.nthreads())
    config   = (hardware = hardware,)

    nu = 1e-5
    U0 = [10.0, 0.0, 0.0]

    BCs = assign(region=mesh, (
        U     = [Dirichlet(:inlet, U0), Extrapolated(:outlet), Wall(:wall, [0.0, 0.0, 0.0]), Extrapolated(:top)],
        p     = [Extrapolated(:inlet), Dirichlet(:outlet, 0.0), Extrapolated(:wall), Extrapolated(:top)],
        k     = [Dirichlet(:inlet, 1.0), Extrapolated(:outlet), KWallFunction(:wall), Extrapolated(:top)],
        omega = [Dirichlet(:inlet, 1000.0), Extrapolated(:outlet), OmegaWallFunction(:wall), Extrapolated(:top)],
        nut   = [Extrapolated(:inlet), Extrapolated(:outlet), NutWallFunction(:wall), Extrapolated(:top)],
    ))
    kwall = only(bc for bc in BCs.k if bc isa KWallFunction)
    wall_faces = kwall.IDs_range
    wall_cells = [mesh.boundary_cellsID[f] for f in wall_faces]

    # Put every wall cell well onto the LOG branch. The production is only
    # written for y+ > yPlusLam; below it both densities return 0 and the ratio
    # below would pass vacuously. Sized for y+ = 50 at the closest wall face.
    (; cmu, yPlusLam) = kwall.value
    delta_min = minimum(mesh.faces[f].delta for f in wall_faces)
    k0 = (50*nu/(cmu^0.25*delta_min))^2

    function wall_production(rho)
        model = Physics(
            time       = Steady(),
            fluid      = Fluid{Incompressible}(nu=nu, rho=rho),
            turbulence = RANS{KOmega}(),
            energy     = Energy{Isothermal}(),
            domain     = mesh,
        )
        initialise!(model.momentum.U, U0)
        initialise!(model.turbulence.k, k0)
        initialise!(model.turbulence.nut, k0/1000.0)
        P = ScalarField(mesh)
        XCALibre.ModelPhysics.set_production!(P, kwall, model, nothing, config)
        return P.values[wall_cells]
    end

    @test minimum(cmu^0.25*mesh.faces[f].delta*sqrt(k0)/nu for f in wall_faces) > yPlusLam

    P_unit  = wall_production(1.0)
    P_dense = wall_production(1000.0)

    # Non-vacuous: the unit-density production is genuinely non-zero ...
    @test all(>(0), P_unit)
    # ... and scales exactly with rho. Without the fix the two are identical.
    @test P_dense ≈ 1000.0 .* P_unit rtol=1e-12
    @test !(P_dense ≈ P_unit)
end
