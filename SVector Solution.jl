using CUDA
using FVM_1D
using StaticArrays
using Adapt

# load existing mesh
# quad and trig 40 and 100
mesh_file = "unv_sample_meshes/trig40.unv"
mesh = build_mesh(mesh_file, scale=0.001)

meshgpu = cu(mesh.cells)

cell = mesh.cells[1]

cell.nodesID
TestCell(
    cell.nodesID,
    cell.facesID,
    cell.neighbours,
    cell.nsign,
    cell.centre,
    cell.volume
    )

abstract type AbstractMockMesh end

#Mesh structure changes
struct TestCell{I,F} # say 60 for now?
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

struct MockMesh{C} <: AbstractMockMesh
    cells::C
end
Adapt.@adapt_structure TestVector

function TestCell(I, F)
    zf = zero(F)
    vec3F = SVector{3, F}(zf, zf, zf)
    TestCell(SVector(rand(I, 24)...), SVector(rand(I, 24)...), SVector(rand(I, 24)...),SVector(rand(I, 24)...),
    vec3F, zf)
end

N = 24
I = Int8
F = Float32

cells = [TestCell(I, F) for i in 1:10]

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

