#=
    potential changes:
    - change TestCell struct to include CuArrays instead of SVectors since the function below fills the zeros as necessary
    - Implement method to generlise number of elements in TestCell arrays
=#

using CUDA
using FVM_1D
using StaticArrays
using Adapt

abstract type AbstractMockMesh end

# load existing mesh
# quad and trig 40 and 100
mesh_file = "unv_sample_meshes/trig40.unv"
mesh = build_mesh(mesh_file, scale=0.001)

struct MockMesh{C} <: AbstractMockMesh
    cells::C
end
Adapt.@adapt_structure MockMesh

## Method 1 struct
struct TestCell{I,F} # say 60 for 3D?
    nnodes::I
    nfaces::I
    nneighbours::I
    nodesID::SVector{24,I}
    facesID::SVector{24,I}
    neighbours::SVector{24,I}
    nsign::SVector{24,I}
    centre::SVector{3, F}
    volume::F
end
Adapt.@adapt_structure TestCell

## Method 1 - change types of mesh fields to SVector
testcell = TestCell{Int64, Float64}[]
for i in 1:length(mesh.cells)

    # arrays
    nodesID = SVector(mesh.cells[i].nodesID...,zeros(Int64,24-length(mesh.cells[i].nodesID))...)
    facesID = SVector(mesh.cells[i].facesID...,zeros(Int64,24-length(mesh.cells[i].facesID))...)
    neighbours = SVector(mesh.cells[i].neighbours...,zeros(Int64,24-length(mesh.cells[i].neighbours))...)
    nsign = SVector(mesh.cells[i].nsign...,zeros(Int64,24-length(mesh.cells[i].nsign))...)
    centre = SVector(mesh.cells[i].centre)

    # numbers
    nnodes = length(nodesID[nodesID .> 0])
    nfaces = length(facesID[facesID .> 0])
    nneighbours = length(neighbours[neighbours .> 0])
    volume = mesh.cells[i].volume

    push!(testcell, TestCell(nnodes, nfaces, nneighbours, nodesID, facesID, neighbours, nsign, centre, volume))
end

# implement code to change SVectors to CuArrays as necessary



## Method 2 struct

struct TestCellGPU{I,F} # say 60 for 3D?
    nnodes::I
    nfaces::I
    nneighbours::I
    nodesID::CuArray{I, 1, CUDA.Mem.DeviceBuffer}
    facesID::CuArray{I, 1, CUDA.Mem.DeviceBuffer}  
    neighbours::CuArray{I, 1, CUDA.Mem.DeviceBuffer}
    nsign::CuArray{I, 1, CUDA.Mem.DeviceBuffer}
    centre::CuArray{F, 1, CUDA.Mem.DeviceBuffer}
    volume::F
end
Adapt.@adapt_structure TestCellGPU

## Method 2

testcell = TestCellGPU{Int64, Float64}[]
for i in 1:length(mesh.cells)

    # arrays
    nodesID = CuArray([mesh.cells[i].nodesID...,zeros(Int64,24-length(mesh.cells[i].nodesID))...])
    facesID = CuArray([mesh.cells[i].facesID...,zeros(Int64,24-length(mesh.cells[i].facesID))...])
    neighbours = CuArray([mesh.cells[i].neighbours...,zeros(Int64,24-length(mesh.cells[i].neighbours))...])
    nsign = CuArray([mesh.cells[i].nsign...,zeros(Int64,24-length(mesh.cells[i].nsign))...])
    centre = CuArray(mesh.cells[i].centre)

    # numbers
    nnodes = length(nodesID[nodesID .> 0])
    nfaces = length(facesID[facesID .> 0])
    nneighbours = length(neighbours[neighbours .> 0])
    volume = mesh.cells[i].volume

    push!(testcell, TestCellGPU(nnodes, nfaces, nneighbours, nodesID, facesID, neighbours, nsign, centre, volume))
end

testcell[1].nodesID

## Kernel
aCPU = test.TestCells[1].facesID
a = CuArray(aCPU)
b = CUDA.zeros(Int8,length(a))

function testKernel!(a,b) # check order
    i = threadIdx().x
    @inbounds if i <= length(b) && i > 0
        b[i] = a[i] + a[i]
    end
    return nothing
end
@cuda threads = length(b) testKernel!(a,b)
a
b

@kernel function double_nodesID2!(A) # I wouldn't want to do this in reality :-)
    I = @index(Global)
    nodesID = A[I].nodesID
    # nids = @private Int32 length(nodesID)
    # for (i, id) ∈ enumerate(nodesID)
    #     nids[i] = id
    # end
    # A[I] = Test0(A[I].centre, nids)
    for id ∈ nodesID
        @print(id,"\n")
    end
end

