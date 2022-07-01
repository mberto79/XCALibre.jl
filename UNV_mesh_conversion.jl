using Plots
using StaticArrays
using BenchmarkTools

using FVM_1D

using FVM_1D.Mesh2D
using FVM_1D.UNV

points, elements, boundaries = load("unv_sample_meshes/quad.unv", Int64, Float64)

@time nodes = build_mesh("unv_sample_meshes/quad.unv")

fig = scatter()
for point ∈ points 
    xyz = point.xyz 
    x = xyz[1]
    y = xyz[2]
    z = xyz[3]
    scatter!(fig, [x],[y], color=:blue, legend=false)
end

@show fig

struct Test0
    a::Vector{Int64}
    b::Vector{Float64}
end

t0 = Test0([],[])

for i ∈ 1:100
    push!(t0.a, i)
    push!(t0.b, i + 0.15)
end
t0.b

a = [[1,2,3],[4,5],[6,7,8,9,10]]

a[2]

struct Face
    nodes::Vector{Int64}
    faces::Vector{Int64}
    centre::SVector{3,Float64}
end

struct Faces
    nodes::Vector{Vector{Int64}}
    faces::Vector{Vector{Int64}}
    centre::Vector{SVector{3,Float64}}
end

faces_individual = Face[]

nodes = Vector{Int64}[]; faces = Vector{Int64}[]; centres = SVector{3,Float64}[]
for i ∈ 1:32000
    push!(faces_individual, Face([i,i,i],[i,i,i],SVector{3,Float64}(i+0.1,i+0.2,i+0.3)))
    push!(nodes, [i,i,i])
    push!(faces, [i,i,i])
    push!(centres, SVector{3,Float64}(i+0.1,i+0.2,i+0.3))
end
nodes
faces_vector = Faces(nodes,faces,centres)


faces_individual[1].centre

faces_vector.centre[1]

function test_individual(faces::Vector{Face})
    nfaces = length(faces)
    sum = SVector{3,Float64}(0.0,0.0,0.0)
    for i ∈ 1:length(faces)
        face = faces[i]
        sum += face.centre 
    end
    average = sum/nfaces        
end


function test_vector(faces::Faces)
    centres = faces.centre
    nfaces = length(centres)
    sum = SVector{3,Float64}(0.0,0.0,0.0)
    for i ∈ 1:length(centres)
        sum += centres[i]
    end
    average = sum/nfaces 
end

@benchmark test_individual($faces_individual)
@benchmark test_vector($faces_vector)