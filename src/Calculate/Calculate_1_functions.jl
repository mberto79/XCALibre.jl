export interpolate!
export nonorthogonal_correction!, correct!

# Scalar face interpolation

function interpolate!(::Type{Linear}, phif::FaceScalarField{I,F}, phi, BCs) where {I,F}
    nbfaces = total_boundary_faces(phi.mesh)
    start = nbfaces + 1
    (; mesh, values) = phi
    (; cells, faces, boundaries) = mesh
    for fi ∈ start:length(faces)
        (; ownerCells) = faces[fi]
        w, df = weight(Linear, cells, faces, fi)
        cID1 = ownerCells[1]
        cID2 = ownerCells[2]
        phi1 = values[cID1]
        phi2 = values[cID2]
        phif.values[fi] = w*phi1 + (1.0 - w)*phi2
    end
    # boundary faces
    for BC ∈ BCs
        bi = boundary_index(mesh, BC.name)
        boundary = mesh.boundaries[bi]
        correct_boundary!(BC, phif, phi, boundary, faces)
    end
end

function correct_boundary!(
    BC::Dirichlet, phif::FaceScalarField{I,F}, phi, boundary, faces) where {I,F}
    (;facesID, cellsID) = boundary
    for fID ∈ facesID
        phif.values[fID] = BC.value 
    end
end

function correct_boundary!(
    BC::Neumann, phif::FaceScalarField{I,F}, phi, boundary, faces) where {I,F}
    (;facesID, cellsID) = boundary
    for fi ∈ eachindex(facesID)
        fID = facesID[fi]
        cID = cellsID[fi]
        (; normal, e, delta) = faces[fID]
        phif.values[fID] = phi.values[cID] + BC.value*delta*(normal⋅e)
    end
end

function correct_interpolation!(
    ::Type{Linear}, phif::FaceScalarField{I,F}, grad, phif0) where {I,F}
    mesh = phif.mesh
    start = total_boundary_faces(mesh) + 1
    (; cells, faces) = mesh
    for fi ∈ start:length(faces)
        (; ownerCells) = faces[fi]
        w, df = weight(Linear, cells, faces, fi)
        cID1 = ownerCells[1]
        cID2 = ownerCells[2]
        grad1 = grad(cID1)
        grad2 = grad(cID2)
        grad_ave = w*grad1 + (1.0 - w)*grad2
        phif.values[fi] = phif0[fi] + grad_ave⋅df
    end
end

# Gradient face interpolation

function interpolate!(::Type{Linear}, gradf::FaceVectorField{I,F}, grad, BCs) where {I,F}
    (; mesh, x, y, z) = gradf
    (; cells, faces) = mesh
    nbfaces = total_boundary_faces(mesh)
    start = nbfaces + 1
    for fi ∈ start:length(faces)
        (; ownerCells) = faces[fi]
        w, df = weight(Linear, cells, faces, fi)
        cID1 = ownerCells[1]
        cID2 = ownerCells[2]
        grad1 = grad(cID1)
        grad2 = grad(cID2)
        gradi = w*grad1 + (1.0 - w)*grad2
        x[fi] = gradi[1]
        y[fi] = gradi[2]
        z[fi] = gradi[3]
    end
    # boundary faces
    for BC ∈ BCs
        bi = boundary_index(mesh, BC.name)
        boundary = mesh.boundaries[bi]
        correct_boundary!(BC, gradf, grad, boundary, faces)
    end
end

function correct_boundary!( # Another way is to use the boundary value and geometry to calc
    BC::Dirichlet, gradf::FaceVectorField{I,F}, grad, boundary, faces) where {I,F}
    (; mesh, x, y, z) = gradf
    (; facesID) = boundary
    for fID ∈ facesID
        face = faces[fID]
        normal = faces[fID].normal
        cID = face.ownerCells[1]
        grad_cell = grad(cID)
        grad_boundary = grad_cell #.*normal .+ grad_cell
        x[fID] = grad_boundary[1]
        y[fID] = grad_boundary[2]
        z[fID] = grad_boundary[3]
    end
end

function correct_boundary!(
    BC::Neumann, gradf::FaceVectorField{I,F}, grad, boundary, faces) where {I,F}
    (; mesh, x, y, z) = gradf
    (; facesID) = boundary
    for fID ∈ facesID
        face = faces[fID]
        normal = faces[fID].normal
        cID = face.ownerCells[1]
        grad_cell = grad(cID)
        grad_boundary =   grad_cell # needs sorting out!
        x[fID] = grad_boundary[1]
        y[fID] = grad_boundary[2]
        z[fID] = grad_boundary[3]
    end
end

function correct_interpolation!(::Type{Linear}, gradf, grad, phi)
    mesh = grad.mesh
    values = phi.values
    start = total_boundary_faces(mesh) + 1
    (; cells, faces) = mesh
    for fi ∈ start:length(faces)
        (; ownerCells, delta) = faces[fi]
        w, df = weight(Linear, cells, faces, fi)
        cID1 = ownerCells[1]
        cID2 = ownerCells[2]
        c1 = cells[cID1].centre
        c2 = cells[cID2].centre
        distance = c2 - c1
        d = distance/delta
        grad1 = grad(cID1)
        grad2 = grad(cID2)
        grad_ave = w*grad1 + (1.0 - w)*grad2
        grad_corr = grad_ave + ((values[cID2] - values[cID1])/delta - (grad_ave⋅d))*d
        gradf.x[fi] = grad_corr[1]
        gradf.y[fi] = grad_corr[2]
        gradf.z[fi] = grad_corr[3]
    end
end

### Non-orthogonality correction (Laplacian terms)

function correct!(eqn::Equation{I,F}, term, non_flux::FaceScalarField{I,F}) where {I,F}
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
    for i ∈ 1:2
        cellsID = mesh.boundaries[i].cellsID
        facesID = mesh.boundaries[i].facesID
        for fi ∈ eachindex(facesID)
            fID = facesID[fi]
            cID = cellsID[fi]
            b[cID] += -sign*J*non_flux.values[fID]
        end
    end
end

function nonorthogonal_correction!(
    tgrad::Grad{S,I,F}, gradf::FaceVectorField{I,F}, phif::FaceScalarField{I,F}
    ) where {S,I,F}
    (; phi) = tgrad
    grad!(tgrad, phif, phi)
    interpolate!(get_scheme(tgrad), gradf, tgrad)
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