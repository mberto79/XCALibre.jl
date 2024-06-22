using FVM_1D
using FVM_1D.FoamMesh
using StaticArrays

mesh_file = "unv_sample_meshes/OF_cavity_hex/constant/polyMesh"

points, face_nodes, face_neighbour_cell, face_owner_cell, bnames, bnFaces, bstartFace = FVM_1D.FoamMesh.read_foamMesh(mesh_file; integer=Int64, float=Float64)

@time fowners = connect_mesh(points, face_nodes, face_neighbour_cell, face_owner_cell, bnames, bnFaces, bstartFace, Int64, Float64)

fowners[2]

a = findall(==(1), fowners[1])
b = findall(==(1), fowners[2])

a = Vector{Int}[]

push!(a, [1,2,3])
a[1][3] = 5
a

for i ∈ eachindex(a)
    println(i)
end

fowners[2]
