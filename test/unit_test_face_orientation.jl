using XCALibre
using LinearAlgebra
using StaticArrays

# Face normals must point out of the owner cell (owner to neighbour). The orientation used to
# be decided by testing each normal against an estimated cell centre, which fails on skewed or
# concave cells where the estimate lies on the wrong side of a face (10M-cell motorBike mesh: 92
# cells whose faces did not close). It is now decided from each cell's topology.

import XCALibre.Mesh: _outward_edge_signs, _outward_face_signs

# ---- 2D: a concave (dart) cell whose node average lies outside it ------------------------
dart = [SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0), SVector(0.25, 0.25, 0.0), SVector(0.0, 1.0, 0.0)]
node_average = sum(dart)/4
edges_ccw = [(1, 2), (2, 3), (3, 4), (4, 1)]
for flips in ([false, false, false, false], [true, false, true, false], [false, true, true, true])
    edges = [f ? (b, a) : (a, b) for ((a, b), f) in zip(edges_ccw, flips)]
    signs, ok = _outward_edge_signs(dart, edges)
    @test ok
    @test signs == [f ? -1 : 1 for f in flips]
end
# the old centre test takes the edge from node 2 to node 3 to point inwards
t = dart[3] - dart[2]; outward = SVector(t[2], -t[1], 0.0)
@test ((dart[2] + dart[3])/2 - node_average) ⋅ outward < 0

# ---- 3D: cell 80259 of the motorBike 0.6M mesh, faces in outward node order --------------
# Faces 4-6 are boundary faces; the old test against the average of the face centres flipped 6.
points = [
    SVector(0, 0, 0),
    SVector(-0.0089874969104024593, 0.0027923340516830264, 0.0025211030796956635),
    SVector(-0.017047568841352723, 0.0042494096300187689, 0.0015223759293256922),
    SVector(-0.014864302935261531, -0.0028370856123865473, 0.0010630023523594545),
    SVector(-0.016525826013901179, 0.0013461526614649832, -0.013549063025014907),
    SVector(-0.010302348133948369, -4.9767604251932385e-05, -0.019751854479047148),
    SVector(-0.0012814099483096086, -0.0017374229369589433, -0.016006173579100391),
    SVector(-0.0088785108228079945, 8.5875796695611406e-05, -0.0079105051097719814),
]
outward_faces = [[1, 2, 3, 4], [3, 5, 6, 4], [7, 1, 4, 6], [7, 8, 2, 1], [2, 8, 5, 3], [7, 6, 5, 8]]
nodes = [(coords = p,) for p in points]
geometry(f) = XCALibre.Mesh.face_geometry(nodes, f, sum(points[f])/length(f))
for flips in (falses(6), BitVector([0, 1, 0, 1, 1, 0]), trues(6))
    fnodes = [fl ? reverse(f) : f for (f, fl) in zip(outward_faces, flips)]
    geo = geometry.(fnodes)
    signs, ok = _outward_face_signs(fnodes, [g[1]*g[2] for g in geo], [g[3] for g in geo])
    @test ok
    @test signs == [fl ? -1 : 1 for fl in flips]
end
geo = geometry.(outward_faces)
centre_estimate = sum(g[3] for g in geo)/6
@test (geo[6][3] - centre_estimate) ⋅ geo[6][1] < 0 # why the old test flipped face 6

# ---- UNV2D_mesh end to end: 3x3 unit grid with one dart cell ------------------------------
function check_2d_mesh(mesh)
    (; cells, cell_faces, cell_nsign, faces, boundary_cellsID) = mesh
    S = [zero(faces[1].normal) for _ in cells]
    nsign_ok = true
    for c in eachindex(cells), k in cells[c].faces_range
        f = cell_faces[k]
        S[c] += cell_nsign[k]*faces[f].normal*faces[f].area
        nsign_ok &= cell_nsign[k] == (faces[f].ownerCells[1] == c ? 1 : -1)
    end
    for (f, c) in enumerate(boundary_cellsID)
        S[c] += faces[f].normal*faces[f].area
    end
    return maximum(norm, S), nsign_ok, sum(cell.volume for cell in cells)
end
unv = read(pkgdir(XCALibre, "examples", "0_GRIDS", "laplace_unit_3by3.unv"), String)
old_node14 = "   3.3333333333333337E-01   3.3333333333333331E-01   0.0000000000000000E+00"
@test count(old_node14, unv) == 1
dart_file = joinpath(mktempdir(), "laplace_unit_3by3_dart.unv")
write(dart_file, replace(unv, old_node14 => "   8.0000000000000002E-02   8.0000000000000002E-02   0.0000000000000000E+00"))
closure, nsign_ok, total_area = check_2d_mesh(UNV2D_mesh(dart_file))
@test closure < 1e-12
@test nsign_ok
@test total_area ≈ 1.0

# ---- compute_3d_geometry! end to end: faces in arbitrary node order are reoriented ---------
# (FOAM3D_mesh keeps OpenFOAM's orientation; UNV3D_mesh builds faces in arbitrary order and
# relies on this.)
mesh = FOAM3D_mesh(pkgdir(XCALibre, "examples", "0_GRIDS", "OF_pitzDaily", "polyMesh"), scale=1)
reference_normals = [face.normal for face in mesh.faces]
reversed = 1:7:length(mesh.faces)
for f in reversed
    reverse!(@view mesh.face_nodes[mesh.faces[f].nodes_range])
end
XCALibre.Mesh.compute_3d_geometry!(mesh; orient_faces=true)
@test all(isapprox(face.normal, n; atol=1e-12) for (face, n) in zip(mesh.faces, reference_normals))
