module Mesh2D
export generate!

using StaticArrays
using LinearAlgebra
using Setfield

include("Mesh2D_0_mesh_types.jl")
include("Mesh2D_1_builder_types.jl")
include("Mesh2D_2_builder.jl")
include("Mesh2D_3_connectivity.jl")
include("Mesh2D_4_geometry.jl")

function generate!(builder::MeshBuilder2D{I,F}) where {I,F}
    GC.gc()
    @time mesh = generate_mesh!(builder)
    println("Number of cells: ", length(mesh.cells))
    mesh
end

function generate_mesh!(builder::MeshBuilder2D{I,F}) where {I,F}
    mesh = build!(builder)
    connect!(mesh, builder)
    geometry!(mesh)
    mesh
end

end