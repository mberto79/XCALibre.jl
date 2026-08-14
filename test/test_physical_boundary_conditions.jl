@testset "physical vector boundary diffusion coefficients" begin
    mesh = FOAM3D_mesh(
        joinpath(pkgdir(XCALibre), "test", "grids", "OF_cavity_hex", "polyMesh"),
        scale=0.001,
    )
    U = VectorField(mesh)
    initialise!(U, [2.0, 3.0, 4.0])
    term = Laplacian{Linear}(ConstantScalar(2.0), U)

    fID = first(mesh.boundaries[1].IDs_range)
    cID = mesh.boundary_cellsID[fID]
    original = mesh.faces[fID]
    inverse_sqrt_two = inv(sqrt(2.0))
    normal = SVector(inverse_sqrt_two, inverse_sqrt_two, 0.0)
    face = Face3D(
        original.nodes_range,
        original.ownerCells,
        original.centre,
        normal,
        original.e,
        original.area,
        original.delta,
        original.weight,
    )
    wall_value = SVector(0.5, -0.5, 1.0)
    wall = Wall(1, wall_value, fID:fID)
    fixed = Dirichlet(1, wall_value, fID:fID)
    slip = Slip(1, 0.0, fID:fID)
    symmetry = Symmetry(1, 0.0, fID:fID)
    empty_indices = Int[]
    empty_values = Float64[]

    for component in (XDir(), YDir(), ZDir())
        arguments = (
            term,
            empty_indices,
            empty_indices,
            empty_values,
            cID,
            cID,
            mesh.cells[cID],
            face,
            fID,
            1,
            component,
            0.0,
        )
        @test wall(arguments...) == fixed(arguments...)
        @test slip(arguments...) == symmetry(arguments...)
    end

    Uf = FaceVectorField(mesh)
    XCALibre.Discretise.boundary_interpolation!(slip, Uf, U, mesh.boundary_cellsID, 0.0, fID)
    actual_normal = original.normal
    @test Uf[fID] ≈ U[cID] - dot(U[cID], actual_normal)*actual_normal
    @test dot(Uf[fID], actual_normal) ≈ 0.0 atol=10eps(Float64)

    XCALibre.Discretise.boundary_interpolation!(symmetry, Uf, U, mesh.boundary_cellsID, 0.0, fID)
    @test Uf[fID] ≈ U[cID] - dot(U[cID], actual_normal)*actual_normal
    @test dot(Uf[fID], actual_normal) ≈ 0.0 atol=10eps(Float64)

    phi = ScalarField(mesh)
    initialise!(phi, 7.0)
    phif = FaceScalarField(mesh)
    XCALibre.Discretise.boundary_interpolation!(
        symmetry, phif, phi, mesh.boundary_cellsID, 0.0, fID)
    @test phif[fID] == phi[cID]

    mdot = FaceScalarField(mesh)
    for scheme in (Linear, Upwind, LUST, BoundedUpwind), flux in (-3.0, 3.0)
        mdot[fID] = flux
        vector_term = Divergence{scheme}(mdot, U)
        scalar_term = Divergence{scheme}(mdot, phi)

        for component in (XDir(), YDir(), ZDir())
            vector_arguments = (
                vector_term,
                empty_indices,
                empty_indices,
                empty_values,
                cID,
                cID,
                mesh.cells[cID],
                face,
                fID,
                1,
                component,
                0.0,
            )
            symmetry_coefficients = symmetry(vector_arguments...)
            @test symmetry_coefficients == slip(vector_arguments...)

            ac, su = symmetry_coefficients
            vc = U[cID]
            vp = vc - dot(vc, normal)*normal
            ap = flux
            expected_residual = scheme === BoundedUpwind ?
                ap*(vp[component.value] - vc[component.value]) :
                ap*vp[component.value]
            @test ac*vc[component.value] - su ≈ expected_residual
        end


        scalar_arguments = (
            scalar_term,
            empty_indices,
            empty_indices,
            empty_values,
            cID,
            cID,
            mesh.cells[cID],
            face,
            fID,
            1,
            nothing,
            0.0,
        )
        scalar_coefficients = symmetry(scalar_arguments...)
        @test scalar_coefficients == slip(scalar_arguments...)
        ac, su = scalar_coefficients
        expected_residual = scheme === BoundedUpwind ? 0.0 : flux*phi[cID]
        @test ac*phi[cID] - su ≈ expected_residual
    end
end
