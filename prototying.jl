using FVM_1D

mesh_file = "unv_sample_meshes/cylinder_d10mm_5mm.unv"
mesh = build_mesh(mesh_file, scale=0.001)

phi = ScalarField(mesh)
J = FaceScalarField(mesh)

J.values .= 2.5

grad = ScalarField(mesh)

model = (
    # Divergence{Linear}(J, phi) 
    # + 
    Laplacian{Linear}(J, phi) 
    == 
    Source(grad)
    # +
    # Source{Linear}(0)
)


eqn = Equation(mesh)
@time discretise!(eqn, model)