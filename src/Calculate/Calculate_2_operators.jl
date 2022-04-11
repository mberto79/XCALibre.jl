export grad!, div! 

function grad!(grad::Grad{Linear,I,F}, phi) where {I,F}
    (; phif, correctors) = grad
    (; mesh) = phi
    (; cells, faces) = mesh
    interpolate!(get_scheme(grad), phif, phi)
    res = [SVector{3,F}(0.0,0.0,0.0) for _ ∈ eachindex(cells)]
    for ci ∈ eachindex(cells)
        (; facesID, nsign, volume) = cells[ci]
        for fi ∈ eachindex(facesID)
            fID = facesID[fi]
            (; area, normal) = faces[fID]
            res[ci] += phif.values[fID]*(area*normal*nsign[fi])
        end
        res[ci] /= volume
    end
    res
end

function div!()
    nothing
end