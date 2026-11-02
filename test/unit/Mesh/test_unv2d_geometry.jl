using XCALibre
using LinearAlgebra
using StaticArrays

# UNV2D cell centres are area centroids, and the face geometry is built from those final centres.

# polygon area centroid from the ordered cell nodes (shoelace)
function polygon_centroid(ps)
    a, cx, cy = 0.0, 0.0, 0.0
    for i in eachindex(ps)
        p, q = ps[i], ps[mod1(i + 1, length(ps))]
        w = p[1]*q[2] - q[1]*p[2]
        a += w; cx += (p[1] + q[1])*w; cy += (p[2] + q[2])*w
    end
    return SVector(cx/(3a), cy/(3a), 0.0), abs(a)/2
end

for grid in ("trig100.unv", "quad40.unv", "cylinder_d10mm_5mm.unv")
    mesh = UNV2D_mesh(pkgdir(XCALibre, "examples", "0_GRIDS", grid), scale=0.001)
    (; cells, faces, nodes, cell_nodes, boundary_cellsID) = mesh
    h = sqrt(sum(c.volume for c in cells)/length(cells)) # typical cell size

    refs = [polygon_centroid([nodes[n].coords for n in cell_nodes[c.nodes_range]]) for c in cells]
    @test all(norm(c.centre - r[1]) <= 1e-8h for (c, r) in zip(cells, refs))
    @test all(isapprox(c.volume, r[2]; rtol=1e-8) for (c, r) in zip(cells, refs))

    nb = length(boundary_cellsID)
    centre(f, i) = cells[f.ownerCells[i]].centre
    internal = (faces[i] for i in nb+1:length(faces))
    @test all(isapprox(f.delta, norm(centre(f, 2) - centre(f, 1)); rtol=1e-10) for f in internal)
    @test all(f.e ≈ (centre(f, 2) - centre(f, 1))/norm(centre(f, 2) - centre(f, 1)) for f in internal)
    boundary = (faces[i] for i in 1:nb)
    d(f) = f.centre - centre(f, 1)
    @test all(f.e ≈ d(f)/norm(d(f)) for f in boundary)
    @test all(d(f)⋅f.normal - 1e-12 <= f.delta <= norm(d(f)) + 1e-12 for f in boundary)
end
