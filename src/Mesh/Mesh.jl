module Mesh
export generate!

using StaticArrays
using LinearAlgebra
using Setfield

include("0_mesh_types.jl")
include("Mesh2D/0_types.jl")
include("Mesh2D/1_builder_types.jl")
include("Mesh2D/2_builder.jl")
include("Mesh2D/3_connectivity.jl")
include("Mesh2D/4_geometry.jl")
include("Mesh2D/5_access_functions.jl")
# include("Plotting/0_plotting.jl")

function generate!(builder::MeshBuilder2D{I,F}) where {I,F}
    GC.gc()
    stats = @timed generate_mesh!(builder)
    mesh = stats.value
    println("Mesh information")
    println("")
    println("Number of cells: ", length(mesh.cells))
    println("Number of faces: ", length(mesh.faces))
    println("Number of nodes: ", length(mesh.nodes))
    println("Number of boundaries: ", length(mesh.boundaries))
    println("")
    println("Mesh built in ", stats.time," seconds (allocated ", stats.bytes," bytes)")
    mesh
end

function generate_mesh!(builder::MeshBuilder2D{I,F}) where {I,F}
    mesh = build!(builder)
    connect!(mesh, builder)
    geometry!(mesh)
    mesh
end

end