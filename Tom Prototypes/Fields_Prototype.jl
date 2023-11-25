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

function SF_GPU!(scalarField)
    (; values, mesh, BCs) = scalarField
    values = cu(values)
    BCs = cu(BCs)
    scalarField = ScalarField(values, mesh, BCs)
end

scalarField = SF_GPU!(scalarField)

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

function FSF_GPU!(faceScalarField)
    (; values, mesh) = faceScalarField
    values = cu(values)
    faceScalarField = FaceScalarField(values, mesh)
end

faceScalarField = FSF_GPU!(faceScalarField)

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

## VECTOR FIELD
VF = VectorField(mesh)

function VF_GPU!(VF)
    (; x, y, z, mesh, BCs) = VF
    x = SF_GPU!(x)
    y = SF_GPU!(y)
    z = SF_GPU!(z)
    BCs = cu(BCs)
    VF = VectorField(x, y, z, mesh, BCs)
end

VF = VF_GPU!(VF)

function test_kernel_VF!(VF)
    i = threadIdx().x + (blockIdx().x-1)*blockDim().x

    (; x, y, z) = VF
    (; values) = x
    valuesX = values
    (; values) = y
    valuesY = values
    (; values) = z
    valuesZ = values
    
    @inbounds if i <= length(valuesX) && i > 0


        if values[i] == 0
            valuesX[i] = valuesX[i] + 1
            valuesY[i] = valuesY[i] + 2
            valuesZ[i] = valuesZ[i] + 3
        else
            valuesX[i] = valuesX[i] + valuesX[i]
            valuesY[i] = valuesY[i] + valuesY[i]
            valuesZ[i] = valuesZ[i] + valuesZ[i]
        end

    end

    # (; mesh) = VF

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

VF.mesh.cells[1].volume
VF.x.values
VF.y.values
VF.z.values

@cuda threads = 1024 blocks = cld(length(VF.x.values),1024) test_kernel_VF!(VF)

VF.mesh.cells[1].volume
VF.x.values
VF.y.values
VF.z.values

## FACE VECTOR FIELD
FVF = FaceVectorField(mesh)

function FVF_GPU!(FVF)
    (; x, y, z, mesh) = FVF
    x = FSF_GPU!(x)
    y = FSF_GPU!(y)
    z = FSF_GPU!(z)
    VF = FaceVectorField(x, y, z, mesh)
end

FVF = FVF_GPU!(FVF)

function test_kernel_FVF!(FVF)
    i = threadIdx().x + (blockIdx().x-1)*blockDim().x

    (; x, y, z) = FVF
    (; values) = x
    valuesX = values
    (; values) = y
    valuesY = values
    (; values) = z
    valuesZ = values
    
    @inbounds if i <= length(valuesX) && i > 0


        if values[i] == 0
            valuesX[i] = valuesX[i] + 1
            valuesY[i] = valuesY[i] + 2
            valuesZ[i] = valuesZ[i] + 3
        else
            valuesX[i] = valuesX[i] + valuesX[i]
            valuesY[i] = valuesY[i] + valuesY[i]
            valuesZ[i] = valuesZ[i] + valuesZ[i]
        end

    end

    # (; mesh) = FVF

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

FVF.mesh.cells[1].volume
FVF.x.values
FVF.y.values
FVF.z.values

@cuda threads = 1024 blocks = cld(length(FVF.x.values),1024) test_kernel_FVF!(FVF)

FVF.mesh.cells[1].volume
FVF.x.values
FVF.y.values
FVF.z.values

## TENSOR FIELD

TF = TensorField(mesh)

function TF_GPU!(TF)
    (; xx, xy, xz, yx, yy, yz, zx, zy, zz, mesh) = TF
    xx = SF_GPU!(xx)
    xy = SF_GPU!(xy)
    xz = SF_GPU!(xz)
    yx = SF_GPU!(yx)
    yy = SF_GPU!(yy)
    yz = SF_GPU!(yz)
    zx = SF_GPU!(zx)
    zy = SF_GPU!(zy)
    zz = SF_GPU!(zz)
    VF = TensorField(xx, xy, xz, yx, yy, yz, zx, zy, zz, mesh)
end

TF = TF_GPU!(TF)

function test_kernel_TF!(TF)
    i = threadIdx().x + (blockIdx().x-1)*blockDim().x

    (; xx) = TF
    (; values) = xx
    
    @inbounds if i <= length(values) && i > 0


        if values[i] == 0
            values[i] = values[i] + 1
        else
            values[i] = values[i] + values[i]
        end

    end

    return nothing

end

TF.xx.values

@cuda threads = 1024 blocks = cld(length(TF.xx.values),1024) test_kernel_TF!(TF)

TF.xx.values

