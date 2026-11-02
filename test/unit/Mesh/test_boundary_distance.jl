using XCALibre
using LinearAlgebra
using StaticArrays

# Boundary face delta = distance from the owner cell centre to the nearest point of the face:
# (Cf - C)·n when the normal from the centre meets the face, and still the true distance on
# skewed cells, where (Cf - C)·n collapses (it is also the wall-function distance).

import XCALibre.Mesh: boundary_distance, weight_delta_e

# ---- unit square face in the z = 0 plane ----------------------------------------------------
for TF in (Float64, Float32)
    square = [SVector{3,TF}(0, 0, 0), SVector{3,TF}(1, 0, 0), SVector{3,TF}(1, 1, 0), SVector{3,TF}(0, 1, 0)]
    nodes = [(coords = x,) for x in square]
    nIDs = 1:4
    centre = sum(square)/TF(4)
    dist(p) = boundary_distance(nodes, nIDs, centre, SVector{3,TF}(p...))
    @test dist((0.3, 0.6, 0.25)) ≈ TF(0.25)               # foot on the face: normal distance
    @test dist((0.3, 0.6, -0.25)) ≈ TF(0.25)              # either side
    @test dist((2.0, 0.5, 0.01)) ≈ sqrt(TF(1) + TF(1e-4)) # skewed: distance to the edge x = 1
    @test dist((2.0, 2.0, 0.0)) ≈ sqrt(TF(2))             # in the plane, beyond a corner
    @test dist((0.5, 0.5, 0.0)) == zero(TF)
    @test (@inferred dist((0.3, 0.6, 0.25))) isa TF

    # the boundary delta is that distance, floored at 0.1% of |Cf - C|
    C = SVector{3,TF}(2.0, 0.5, 0.01)
    _, delta, e = weight_delta_e(centre - C, SVector{3,TF}(0, 0, 1), dist(C))
    @test delta ≈ dist(C)
    @test e ≈ (centre - C)/norm(centre - C)
    _, delta0, _ = weight_delta_e(centre - C, SVector{3,TF}(0, 0, 1), zero(TF))
    @test delta0 ≈ TF(0.001)*norm(centre - C)

    # 2D faces (edges)
    a, b = SVector{3,TF}(0, 0, 0), SVector{3,TF}(1, 0, 0)
    @test boundary_distance(a, b, SVector{3,TF}(0.4, 0.3, 0)) ≈ TF(0.3)
    @test boundary_distance(a, b, SVector{3,TF}(-0.3, 0.4, 0)) ≈ TF(0.5)
end

# ---- FOAM3D_mesh: every boundary face of pitzDaily ------------------------------------------
mesh = FOAM3D_mesh(pkgdir(XCALibre, "examples", "0_GRIDS", "OF_pitzDaily", "polyMesh"), scale=1)
(; faces, cells, face_nodes, nodes, boundary_cellsID) = mesh
nb = length(boundary_cellsID)
owner_centre(face) = cells[face.ownerCells[1]].centre
expected(face) = max(boundary_distance(nodes, face_nodes[face.nodes_range], face.centre, owner_centre(face)),
    0.001*norm(face.centre - owner_centre(face)))
@test all(faces[fID].delta ≈ expected(faces[fID]) for fID in 1:nb)
@test all(wall_normal_distance(faces[fID]) == faces[fID].delta for fID in 1:nb)
# between the normal distance (Cf - C)·n and |Cf - C|
@test all((faces[f].centre - owner_centre(faces[f]))⋅faces[f].normal - 1e-12 <= faces[f].delta <=
    norm(faces[f].centre - owner_centre(faces[f])) + 1e-12 for f in 1:nb)
# hexahedral cells whose normal foot lies on the face: unchanged from (Cf - C)·n
onface = count(fID -> isapprox(faces[fID].delta, (faces[fID].centre - cells[faces[fID].ownerCells[1]].centre)⋅faces[fID].normal; rtol=1e-10), 1:nb)
@test onface > 0.9*nb

# ---- UNV2D_mesh: boundary edges ----------------------------------------------------------------
mesh2 = UNV2D_mesh(pkgdir(XCALibre, "examples", "0_GRIDS", "laplace_unit_3by3.unv"))
# boundary edges: positive and no larger than |Cf - C|
bfaces2 = [mesh2.faces[fID] for boundary in mesh2.boundaries for fID in boundary.IDs_range]
@test all(f.delta > 0 && f.delta <= norm(f.centre - mesh2.cells[f.ownerCells[1]].centre) + 1e-12 for f in bfaces2)
@test all(wall_normal_distance(f) == f.delta for f in bfaces2)
