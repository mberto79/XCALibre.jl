using FVM_1D

mesh_file = "unv_sample_meshes/cylinder_d10mm_5mm.unv"
mesh = build_mesh(mesh_file, scale=0.001)

phi = ScalarField(mesh)
J = 4.0

grad = ScalarField(mesh)

model = (
    Divergence{Upwind}(J, phi) 
    + 
    Laplacian{Linear}(2.0, phi) 
    == 
    Source{Constant}(grad)
    # +
    # Source{Linear}(0)
)

model = (
    Divergence{Linear}(J, phi)  
    + 
    Laplacian{Linear}(J, phi) 
    -
    Divergence{Linear}(J, phi)  
)

@generated function test(model)
    nterms = model.parameters[3]
    nsources = model.parameters[4]
    quote
        term1 = model.terms[1]
        term2 = model.terms[2]
        src1 = model.sources
        # println($nterms, " ", $nsources)
        nothing
    end
end

@time test(model)