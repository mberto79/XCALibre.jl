using XCALibre
using KernelAbstractions
using LinearAlgebra
using StaticArrays
using Test

@testset "face-flux reconstruction includes boundary faces" begin
    config = (hardware=Hardware(backend=CPU(), workgroup=256),)
    expected = SVector(2.0, -1.0, 0.5)

    cases = (
        # hex cavity: well conditioned, exercises the determinant tolerance scaling
        (joinpath(pkgdir(XCALibre), "test", "grids", "OF_cavity_hex", "polyMesh"), 0.001),
        # pitzDaily is one cell thick with an empty patch, so m33 is zero without boundary faces
        (joinpath(pkgdir(XCALibre), "examples", "0_GRIDS", "OF_pitzDaily", "polyMesh"), 1.0),
    )

    for (mesh_file, scale) ∈ cases
        mesh = FOAM3D_mesh(mesh_file, scale=scale)

        Uf = FaceVectorField(mesh)
        initialise!(Uf, expected)
        phif = FaceScalarField(mesh)
        flux!(phif, Uf, config)

        U = VectorField(mesh)
        moments = KernelAbstractions.allocate(CPU(), Float64, length(mesh.cells), 9)
        XCALibre.Solvers.reconstruct!(U, phif, moments, config)
        @test maximum(norm(U[cID] - expected) for cID in eachindex(U)) < 1e-12
    end
end
