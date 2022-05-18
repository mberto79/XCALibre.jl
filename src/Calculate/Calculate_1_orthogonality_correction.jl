export nonorthogonal_correction!, correct!

### Non-orthogonality correction (Laplacian terms)

function correct!(eqn::Equation{I,F}, term, non_flux::FaceScalarField{I,F}) where {I,F}
# function correct!(b, term, non_flux::FaceScalarField{I,F}) where {I,F}
    sign = term.sign[1]
    J = term.J
    mesh = non_flux.mesh
    (; faces, cells) = mesh
    (; b) = eqn 
    for ci ∈ eachindex(cells)
        #cell stuff
        cell = cells[ci]
        (; facesID, nsign) = cell
        for fi ∈ eachindex(facesID)
            # face stuff
            fID = facesID[fi]
            b[ci] += -sign*J*non_flux.values[fID]*nsign[fi]
        end
    end
    nbfaces = total_boundary_faces(mesh)
    for fi ∈ 1:nbfaces
        face = faces[fi]
        cID = face.ownerCells[1]
        b[cID] += -sign*J*non_flux.values[fi]
    end
end

function nonorthogonal_correction!(
    tgrad::Grad{S,I,F}, gradf::FaceVectorField{I,F}, phif::FaceScalarField{I,F}, BCs
    ) where {S,I,F}
    (; phi) = tgrad
    grad!(tgrad, phif, phi, BCs)
    interpolate!(get_scheme(tgrad), gradf, tgrad, BCs)
    nonorthogonal_flux!(phif, gradf)
end

# CAREFUL: overwriting phif to store face flux correction (NOT the interpolation of phi)
function nonorthogonal_flux!(phif::FaceScalarField{I,F}, gradf) where{I,F} 
    (; mesh, x, y, z) = gradf
    (; faces, cells) = mesh
    # start = total_boundary_faces(mesh) + 1
    # for fi ∈ start:length(faces)
    for fi ∈ eachindex(faces)
        face = faces[fi]
        (; area, normal, e) = face
        T = (normal-e)*area
        phif.values[fi] = gradf(fi)⋅T
    end
end

### Weight functions

function weight(::Type{Linear}, cells, faces, fi)
    (; ownerCells, centre) = faces[fi]
    cID1 = ownerCells[1]
    cID2 = ownerCells[2]
    c1 = cells[cID1].centre
    c2 = cells[cID2].centre
    c1_f = centre - c1
    c1_c2 = c2 - c1
    q = (c1_f⋅c1_c2)/(c1_c2⋅c1_c2)
    f_prime = c1 - q*(c1 - c2)
    w = norm(c2 - f_prime)/norm(c2 - c1)
    df = centre - f_prime
    return w, df
end