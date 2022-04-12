export grad!, div! 

function grad!(grad::Grad{Linear,I,F}, phif, phi) where {I,F}
    (; x, y, z, correctors) = grad
    (; mesh) = phi
    (; cells, faces) = mesh

    interpolate!(get_scheme(grad), phif, phi)
    
    for ci ∈ eachindex(cells)
        (; facesID, nsign, volume) = cells[ci]
        res = SVector{3,F}(0.0,0.0,0.0)
        for fi ∈ eachindex(facesID)
            fID = facesID[fi]
            (; area, normal) = faces[fID]
            res += phif.values[fID]*(area*normal*nsign[fi])
        end
        res /= volume
        x[ci] = res[1]
        y[ci] = res[2]
        z[ci] = res[3]
    end
end

function div!()
    nothing
end