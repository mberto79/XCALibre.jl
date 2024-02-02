using Plots
using FVM_1D
using Krylov
using CUDA

# quad, backwardFacingStep_2mm, backwardFacingStep_10mm, trig40
mesh_file = "unv_sample_meshes/cylinder_d10mm_5mm.unv"
mesh = build_mesh(mesh_file, scale=0.001)
mesh = update_mesh_format(mesh)
mesh = cu(mesh)
model = RANS{Laminar}(mesh=mesh, viscosity=ConstantScalar(nu))

BCs = ( 
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Dirichlet(:cylinder, noSlip),
    Neumann(:bottom, 0.0),
    Neumann(:top, 0.0)
)

for arg in BCs
    # for i in eachindex(BCs)
    #     if model.boundary_info[i].Name == arg.ID
    #         idx = model.boundary_info[i].ID
    #         println("Idx = $idx")
    #         break
    #     end
    # end
    boundary_index_test(model.boundary_info,arg.ID)
end



## INDEXING USING BOUNDARY INFO struct
function boundary_index_test(
    boundaries, name
    )
    for i in eachindex(boundaries)
        if boundaries[i].Name == name
            idx = boundaries[i].ID
            println("Idx = $idx")
            break
        end
    end
end
