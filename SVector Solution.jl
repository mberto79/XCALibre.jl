using CUDA
using StaticArrays
using Adapt

abstract type AbstractTest end

#Mesh structure changes
struct TestCellVector{I,F,N}
    nodesID::SVector{N,I}
    facesID::SVector{N,I}
    neighbours::SVector{N,I}
    nsign::SVector{N,I}
    centre::SVector{3, F}
    volume::F
end
Adapt.@adapt_structure TestCellVector

struct TestVector{I,F,N} <: AbstractTest
    TestCellVectors::Vector{TestCellVector{I,F,N}}
    # TestFaces::Vector{TestFace2D{Int8,Float32}}
end
Adapt.@adapt_structure TestVector

function TestCellVector(I, F)
    zf = zero(F)
    vec3F = SVector{3, F}(zf, zf, zf)
    TestCellVector(SVector(rand(I, 24)...), SVector(rand(I, 24)...), SVector(rand(I, 24)...),SVector(rand(I, 24)...),
    vec3F, zf)
end

N = 24
I = Int8
F = Float32

cells_vector = [TestCellVector(I, F) for i in 1:10]

test = TestVector{I, F, N}(cells_vector)

aCPU = test.TestCellVectors[1].facesID
a = CuArray(aCPU)
b = CUDA.zeros(Int8,length(a))

typeof(test.TestCellVectors[1].facesID)

function TestKernel!(a,b)
    i = threadIdx().x
    @inbounds if i <= length(b) && i > 0
        b[i] = a[i] + a[i]
    end
    return nothing
end


@cuda threads = length(b) TestKernel!(a,b)
a
b