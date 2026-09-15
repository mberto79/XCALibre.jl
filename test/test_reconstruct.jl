@testset "face-flux reconstruction includes boundary faces" begin
    mesh = FOAM3D_mesh(
        joinpath(pkgdir(XCALibre), "test", "grids", "OF_cavity_hex", "polyMesh"),
        scale=0.001,
    )
    config = (hardware=Hardware(backend=CPU(), workgroup=256),)
    expected = SVector(2.0, -1.0, 0.5)

    Uf = FaceVectorField(mesh)
    initialise!(Uf, expected)
    phif = FaceScalarField(mesh)
    flux!(phif, Uf, config)

    U = VectorField(mesh)
    XCALibre.Solvers.reconstruct!(U, phif, config)
    @test maximum(norm(U[cID] - expected) for cID in eachindex(U)) < 1e-12
end

