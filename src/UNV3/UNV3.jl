#= 

UNV 3D Mesh Structure and Implementation Assumptions

1. Topological Assumptions and Mesh Contracts

Element Support: The mesh primarily supports 3D unstructured elements: Tetrahedra (4 nodes), Hexahedra (8 nodes), and Prisms/Wedges (6 nodes).

Face Topologies: Faces are strictly either triangular (3 nodes) or quadrilateral (4 nodes).

Face Ownership: Internal Faces are shared by exactly two cells (designated as Owner 1 and Owner 2). Boundary Faces belong to exactly one cell. Owner 1 is the parent cell, and Owner 2 is mapped to the same cell (or a designated boundary ID) to satisfy data structure requirements.

Face Ordered Tracking: The exact 3D spatial node ordering of faces provided by the UNV elements is natively retained. Note: Faces are temporarily sorted behind the scenes purely to generate invariant keys for dictionary hashing, but the actual mesh memory permanently stores the explicit, ordered sequence.

2. Geometric Assumptions and Calculations

True Geometric Centroids: Face Centroids are computed as area-weighted true geometric centers by sub-triangulating the face around an estimated arithmetic center. Cell Centroids are computed as volume-weighted true geometric centers by summing the centroids of the divergence pyramids formed by the cell's boundary faces and an estimated apex.

Face Area and Normals: Computed by summing the cross products of consecutive edge sub-triangles.

The Normal Vector Contract (Right-Hand Rule): The normal vector of a face must always point outward from Owner 1. During the geometry pipeline, if a parsed face sequence produces a normal pointing inward (towards Owner 1's center), the normal is flipped by multiplying by -1, and the ordered node array for that face is permanently reversed in memory to mathematically satisfy the Right-Hand Rule.

Cell Volumes: Computed using the divergence theorem, specifically by summing the volumes of pyramids formed by the cell's true geometric centroid and its boundary faces.

=#

module UNV3

using StaticArrays
using LinearAlgebra
using Accessors
using Adapt
using Printf
using Statistics

using XCALibre.Mesh

include("UNV3_0_types.jl")
include("UNV3_1_reader.jl")
include("UNV3_2_builder.jl")
include("UNV3_check_connectivity.jl")

end