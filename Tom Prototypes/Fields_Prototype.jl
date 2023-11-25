using Plots
using Krylov
using StaticArrays
using CUDA
using FVM_1D

# backwardFacingStep_2mm, backwardFacingStep_10mm
mesh_file = "unv_sample_meshes/backwardFacingStep_10mm.unv"
unv_mesh = build_mesh(mesh_file, scale=0.001)

mesh = mesh2_from_UNV(unv_mesh)

## SCALAR FIELD
scalarField = ScalarField(mesh)

(; values, mesh, BCs) = scalarField

values = cu(values)
BCs = cu(BCs)

scalarField = ScalarField(values, mesh, BCs)

function test_kernel_SF!(scalarField)
    i = threadIdx().x + (blockIdx().x-1)*blockDim().x

    (; values) = scalarField
    
    @inbounds if i <= length(values) && i > 0

        values[i] = values[i] + values[i]

    end

    # ScalarField(values, scalarField.mesh, scalarField.BCs)

    return nothing

end

scalarField.values
@cuda threads = 1024 blocks = cld(length(scalarField.values),1024) test_kernel_SF!(scalarField)
scalarField.values

## FACE SCALAR FIELD
faceScalarField = FaceScalarField(mesh)

(; values, mesh) = faceScalarField
values = cu(values)
faceScalarField = FaceScalarField(values, mesh)

function test_kernel_FSF!(faceScalarField)
    i = threadIdx().x + (blockIdx().x-1)*blockDim().x

    (; values) = faceScalarField
    
    @inbounds if i <= length(values) && i > 0


        if values[i] == 0
            values[i] = values[i] + 1
        else
            values[i] = values[i] + values[i]
        end

    end

    # (; mesh) = faceScalarField

    # (; cells, cell_nodes, cell_faces, cell_neighbours, cell_nsign) = mesh
    
    # @inbounds if i <= length(cells) && i > 0

    #     (; centre, volume, nodes_range, faces_range) = cells[i]
        
    #     for j in nodes_range

    #     cell_nodes[j] = cell_nodes[j] + cell_nodes[j]

    #     end

    #     for k in faces_range

    #         cell_faces[k] = cell_faces[k]+cell_faces[k]
    #         cell_neighbours[k] = cell_neighbours[k] + cell_neighbours[k]
    #         cell_nsign[k] = cell_nsign[k] + cell_nsign[k]

    #     end

    #     centre = centre + centre
    #     volume = volume + volume

    #     cells[i] = Cell{Int64, Float64}(
    #         centre,
    #         volume,
    #         nodes_range,
    #         faces_range
    #     )
    # end

    # mesh = Mesh2(
    #     cells,
    #     cell_nodes,
    #     cell_faces,
    #     cell_neighbours,
    #     cell_nsign,
    #     mesh.faces,
    #     mesh.face_nodes,
    #     mesh.boundaries,
    #     mesh.nodes,
    # )

    return nothing

end

faceScalarField.mesh.cell_nodes
faceScalarField.mesh.cell_faces
faceScalarField.mesh.cells[1].centre
faceScalarField.values

@cuda threads = 1024 blocks = cld(length(faceScalarField.values),1024) test_kernel_FSF!(faceScalarField)

faceScalarField.mesh.cell_nodes
faceScalarField.mesh.cell_faces
faceScalarField.mesh.cells[1].centre
faceScalarField.values