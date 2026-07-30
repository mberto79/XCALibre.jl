# Validation of the generated K-Site wedge mesh.
#
# Run AFTER generating and meshing:
#     cd examples/0_GRIDS/ksite_wedge
#     julia make_ksite_wedge.jl
#     blockMesh
#     cd ../../..
#     julia --project=. test/unit_test_ksite_wedge_mesh.jl
#
# Skips cleanly if the polyMesh has not been generated yet, so it is safe to
# include from runtests.jl.
#
# The point of this file is the axis: blockMesh collapses the r = 0 block faces
# into prism cells, and a zero-area face whose `delta` is also zero would give
# 0/0 in every snGrad kernel in the multiphase solver. That is checked
# explicitly below, along with the conditioning of the moment matrix that
# `reconstruct!` inverts.

using XCALibre
using Test
using LinearAlgebra
using StaticArrays

KSITE_DIR = pkgdir(XCALibre, "examples/0_GRIDS/ksite_wedge")
# blockMesh writes to <case>/constant/polyMesh; FOAM3D_mesh wants that directory
KSITE_POLYMESH = joinpath(KSITE_DIR, "constant", "polyMesh")

# Geometry, taken directly from Fernandes et al. (2026) Sec. 2: the K-Site tank
# is an ellipsoid of major diameter 2.20 m and minor diameter 1.93 m. Both the
# reported volume (4.89 m^3) and surface area (13.98 m^2) follow from these.
WEDGE_ANGLE = 5.0

A_SEMI = 2.20/2      # 1.100 m equatorial semi-axis
B_SEMI = 1.93/2      # 0.965 m polar semi-axis

V_TANK = 4/3*pi*A_SEMI^2*B_SEMI      # 4.891 m^3 (paper: 4.89)

if !isdir(KSITE_POLYMESH)
    @info """K-Site wedge polyMesh not found - skipping mesh validation.
             Generate it with:
                 cd examples/0_GRIDS/ksite_wedge && julia make_ksite_wedge.jl && blockMesh"""
else

mesh = FOAM3D_mesh(KSITE_POLYMESH, scale=1.0, integer_type=Int64, float_type=Float64)

@testset "K-Site wedge: geometry" begin
    # The wedge is a 5/360 slice of the full tank.
    V_expected = V_TANK*WEDGE_ANGLE/360
    V_mesh = sum(c.volume for c in mesh.cells)

    # blockMesh chords the elliptical arcs, so the meshed volume is slightly
    # under the analytical one; 1% is generous for the default resolution.
    @test isapprox(V_mesh, V_expected; rtol=0.01)

    # every cell must have positive volume (no inverted/degenerate cells)
    @test all(c.volume > 0 for c in mesh.cells)

    # all cell centres inside the ellipsoid (allow a small chording tolerance)
    for c in mesh.cells
        r = hypot(c.centre[1], c.centre[2])
        @test (r/A_SEMI)^2 + (c.centre[3]/B_SEMI)^2 <= 1.02
    end
end

@testset "K-Site wedge: face metrics are finite (axis degeneracy)" begin
    # This is the check that motivated the whole exercise.
    for (i, f) in enumerate(mesh.faces)
        @test isfinite(f.area)
        @test f.area >= 0
        @test isfinite(f.delta)
        @test all(isfinite, f.normal)
        @test all(isfinite, f.centre)
    end

    # A zero-area face is tolerable (it carries no flux, because every kernel
    # multiplies by area) but a zero `delta` is NOT: 1/delta appears in every
    # surface-normal-gradient kernel and would give 0/0.
    min_delta = minimum(f.delta for f in mesh.faces)
    @test min_delta > 0

    # Faces with area carry a unit normal. The zero-area axis faces do not have
    # a normal direction at all, which is legitimate - but they must be exactly
    # the axis faces and nothing else, so check that rather than waving them
    # through.
    degenerate = findall(f -> f.area == 0, mesh.faces)

    for (i, f) in enumerate(mesh.faces)
        if f.area > 0
            @test isapprox(norm(f.normal), 1.0; atol=1e-8)
        end
    end

    # Every degenerate face must sit on the axis (r = 0) ...
    for i in degenerate
        r = hypot(mesh.faces[i].centre[1], mesh.faces[i].centre[2])
        @test isapprox(r, 0.0; atol=1e-12)
    end

    # ... and must all belong to a single empty patch (`axis`, or `defaultFaces`
    # if the mesh was built before the axis patch was named explicitly).
    axis_patch = only(b for b in mesh.boundaries
                      if b.name === :axis || b.name === :defaultFaces)
    @test Set(degenerate) == Set(axis_patch.IDs_range)

    # Topology check: core contributes n_core_z faces, north and south n_rad each.
    @info "axis faces" n=length(degenerate) patch=axis_patch.name
end

@testset "K-Site wedge: boundary patches" begin
    names = [b.name for b in mesh.boundaries]
    @test :tankWall in names
    @test :wedgeFront in names
    @test :wedgeBack in names
    # the axis must be a named patch: `assign` asserts one BC per patch, so an
    # unnamed axis would surface as a confusing count mismatch in the case file
    @test (:axis in names) || (:defaultFaces in names)

    # the two wedge planes must carry the same number of faces
    front = only(b for b in mesh.boundaries if b.name === :wedgeFront)
    back  = only(b for b in mesh.boundaries if b.name === :wedgeBack)
    @test length(front.IDs_range) == length(back.IDs_range)

    # wall area of a 5 deg slice of the ellipsoid: compare against the
    # analytical oblate-spheroid surface area, scaled by the wedge fraction.
    wall = only(b for b in mesh.boundaries if b.name === :tankWall)
    A_wall = sum(mesh.faces[fID].area for fID in wall.IDs_range)

    e = sqrt(1 - (B_SEMI/A_SEMI)^2)                       # oblate spheroid
    S = 2pi*A_SEMI^2 * (1 + (1-e^2)/e * atanh(e))
    @test isapprox(A_wall, S*WEDGE_ANGLE/360; rtol=0.02)
end

@testset "K-Site wedge: reconstruct! recovers a uniform field" begin
    # Exercises the real `reconstruct!` rather than a copy of its kernel.
    #
    # For a uniform cell vector `g`, the exact face data is psif_f = area_f *
    # (g . n_f), so the reconstruction must return `g` in every cell. This is
    # the regression test for the boundary-face omission: `mesh.cell_faces`
    # holds internal faces only, and on a one-cell-thick wedge BOTH wedge faces
    # are boundary faces, which previously left the moment matrix exactly rank 2
    # in every cell and produced an identically zero reconstruction.
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

    @test err < 1e-10

    # The azimuthal component must vanish: the wedge planes are symmetry planes,
    # so an axisymmetric field has no theta component. Including the boundary
    # faces (where psif = 0) is exactly what imposes this.
    @test maximum(abs, phi.y.values) < 1e-12

    # And no cell may come back identically zero, which was the old failure mode.
    n_zero = count(i -> phi.x.values[i] == 0 && phi.z.values[i] == 0,
                   1:length(mesh_dev.cells))
    @test n_zero == 0
end

@testset "K-Site wedge: adiabatic two-phase uniform T is preserved" begin
    # REGRESSION TEST for a bug the rest of the suite structurally could not
    # catch: the energy tests all ran on quad40, and the wedge tests never ran
    # the energy equation.
    #
    # A sealed adiabatic tank with a uniform initial temperature must hold that
    # temperature exactly, no matter what the flow does — there is no source and
    # no flux through any boundary. Buoyancy still stirs the fluid, so this
    # exercises advection of T across the liquid/vapour interface.
    #
    # It failed by 7.9 K in five 1 ms steps because the time term used the
    # CURRENT rho_cp (non-conservative, rho_cp*dT/dt) while the divergence term
    # was conservative, leaving a spurious `T*div(rho_cp_phi)` source. That error
    # scales with (rho*cp)_l - (rho*cp)_v = 6.7e5, so it dominated everything
    # downstream: an 11.6 K interface spike, 14.8 m/s spurious velocities, a
    # collapsing time step, and a pressure equation that Cg rejected as not
    # positive definite. Fixed by passing `rho_prev=rho_cp_prev` to `discretise!`.
    backend = CPU()
    hardware = Hardware(backend=backend, workgroup=AutoTune())
    mesh_dev = adapt(backend, mesh)
    noSlip = [0.0, 0.0, 0.0]
    T0 = 20.43

    model = Physics(
        time = Transient(),
        fluid = Fluid{Multiphase}(
            model = VOF(cAlpha=1.0, sigma=0.0),
            phases = (Phase(rho=70.8, mu=13.2e-6, k=0.100, cp=9660.0),
                      Phase(rho=1.34, mu=1.11e-6, k=0.0169, cp=12200.0)),
            gravity = Gravity([0.0, 0.0, -9.81])),     # gravity ON: the trigger
        turbulence = RANS{Laminar}(),
        energy = Energy{TwoPhaseTemperature}(Tref=T0),
        domain = mesh_dev)

    BCs = assign(region=mesh_dev, (
        U = [Wall(:tankWall, noSlip), Symmetry(:wedgeFront),
             Symmetry(:wedgeBack), Empty(:axis)],
        p_rgh = [Zerogradient(:tankWall), Symmetry(:wedgeFront),
                 Symmetry(:wedgeBack), Empty(:axis)],
        alpha = [Zerogradient(:tankWall), Symmetry(:wedgeFront),
                 Symmetry(:wedgeBack), Empty(:axis)],
        T = [Zerogradient(:tankWall), Symmetry(:wedgeFront),   # adiabatic
             Symmetry(:wedgeBack), Empty(:axis)]))

    s = Schemes(time=Euler, divergence=Upwind, laplacian=Linear, gradient=Gauss)
    b() = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(),
                      convergence=1e-10, relax=1.0, rtol=0.0, atol=1e-14)
    config = Configuration(
        solvers=(U=b(), p_rgh=b(), alpha=b(), T=b()),
        schemes=(U=s, p=s, p_rgh=s, alpha=s, T=s),
        runtime=Runtime(iterations=5, time_step=1.0e-3, write_interval=-1),
        hardware=hardware, boundaries=BCs)

    initialise!(model.momentum.U, noSlip)
    initialise!(model.fluid.p_rgh, 0.0)
    initialise!(model.fluid.alpha, 0.0)
    setField_Box!(mesh=mesh, field=model.fluid.alpha, value=1.0,
                  min_corner=[-1.2, -1.2, -1.0], max_corner=[1.2, 1.2, 0.0])
    initialise!(model.energy.T, T0)

    run!(model, config)

    dev = maximum(abs, model.energy.T.values .- T0)
    @info "adiabatic uniform-T drift on wedge" dev
    @test dev < 1e-10
end

end # polyMesh exists
