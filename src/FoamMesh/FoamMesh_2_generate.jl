export generate_mesh

function generate_mesh(foamdata, connectivity, integer, float)
    boundaries = generate_boundaries(foamdata, integer, float)
end

function generate_boundaries(foamdata, TI, TF)
    (; n_bfaces, n_faces, n_ifaces) = foamdata
    foamBoundaries = foamdata.boundaries
    boundaries = Mesh.Boundary{Symbol,UnitRange{TI}}[
        Mesh.Boundary(:default, UnitRange{TI}(0,0)) for _ ∈ eachindex(foamBoundaries)]

    startIndex = 1
    endIndex = 0
    for (bi, foamboundary) ∈ enumerate(foamBoundaries)
        (; name, nFaces) = foamboundary
        endIndex = startIndex + nFaces - one(TI)
        boundaries[bi] = Mesh.Boundary{Symbol,UnitRange{TI}}(name, startIndex:endIndex)
        startIndex += nFaces
    end

    return boundaries
end