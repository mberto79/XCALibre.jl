using Plots

using FVM_1D

using Krylov

# backwardFacingStep_2mm, backwardFacingStep_10mm
mesh_file = "unv_sample_meshes/backwardFacingStep_10mm.unv"
mesh = build_mesh(mesh_file, scale=0.001)

# p = ScalarField(mesh)
# U = VectorField(mesh)

gradU = Grad{Linear}(U)
Uf = FaceVectorField(mesh)

grad!(gradU, Uf, U, U.BCs)

gradU[20]

gradUT = T(gradU)

gradUT[20]
gradUT.parent.result[20]

gradU.result[20]

using LinearAlgebra

norm(gradU[20])
norm(gradU.result[20])

S = zeros(Float64, length(mesh.cells))

for i ∈ eachindex(S)
    S[i] = norm(gradU[i])
end

surface(x(mesh), y(mesh), S, view_angle=(0,90))

p.values .= S

write_vtk("results", mesh, ("U", U), ("p", p))
