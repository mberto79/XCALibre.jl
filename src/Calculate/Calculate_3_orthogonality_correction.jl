export nonorthogonal_correction!, correct!
export nonorthogonal_flux!

### Non-orthogonality correction (Laplacian terms)

function correct!(eqn::ScalarEquation{I,F}, term, corr_flux::FaceScalarField{I,F}) where {I,F}
    sign = term.sign[1]
    J = term.J
    mesh = corr_flux.mesh
    (; faces, cells) = mesh
    (; b) = eqn 
    (; values) = corr_flux
    for fID ∈ eachindex(faces)
        face = faces[fID]
        owners = face.ownerCells
        cell1 = owners[1]
        cell2 = owners[2]
        correction = -sign*J(fID)*values[fID]
        b[cell1] += correction          # normal aligns with cell1
        b[cell2] += correction*(-1.0)   # correct normal for cell2
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
    start = total_boundary_faces(mesh) + 1
    for fi ∈ start:length(faces)
    # for fi ∈ eachindex(faces)
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