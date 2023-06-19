using FVM_1D

mesh_file = "unv_sample_meshes/cylinder_d10mm_5mm.unv"
mesh = build_mesh(mesh_file, scale=0.001)

phi = ScalarField(mesh)
J = 4.0

grad = ScalarField(mesh)

model = begin
    Divergence{Linear}(J, phi) - Laplacian{Linear}(J, phi) == Source{Constant}(grad)
end