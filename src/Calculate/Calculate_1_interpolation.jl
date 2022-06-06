export interpolate!

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
    correct_interpolation!(Linear, gradf, grad)
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
        # grad_boundary = ((BC.value - grad.phi.values[fID])/face.delta)*normal
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
        grad_boundary =   grad_cell - (grad_cell⋅normal)*normal # needs sorting out!
        x[fID] = grad_boundary[1]
        y[fID] = grad_boundary[2]
        z[fID] = grad_boundary[3]
    end
end

function correct_interpolation!(::Type{Linear}, gradf, grad)
    mesh = grad.mesh
    values = grad.phi.values
    start = total_boundary_faces(mesh) + 1
    (; cells, faces) = mesh
    for fi ∈ start:length(faces)
    # for fi ∈ 1:length(faces)
        (; ownerCells, delta, e) = faces[fi]
        w, df = weight(Linear, cells, faces, fi)
        cID1 = ownerCells[1]
        cID2 = ownerCells[2]
        # c1 = cells[cID1].centre
        # c2 = cells[cID2].centre
        # distance = c2 - c1
        # d = distance/delta
        d = e
        # grad1 = grad(cID1)
        # grad2 = grad(cID2)
        # grad_ave = w*grad1 + (1.0 - w)*grad2
        grad_ave = gradf(fi)
        grad_corr = grad_ave + ((values[cID2] - values[cID1])/delta - (grad_ave⋅d))*d
        gradf.x[fi] = grad_corr[1]
        gradf.y[fi] = grad_corr[2]
        gradf.z[fi] = grad_corr[3]
    end
end

# Vector face interpolation

function interpolate!(gradf::FaceVectorField{I,F}, grad::VectorField{I,F}, BCs) where {I,F}
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
    # correct_interpolation!(Linear, gradf, grad)
    # boundary faces
    for BC ∈ BCs
        bi = boundary_index(mesh, BC.name)
        boundary = mesh.boundaries[bi]
        correct_boundary!(BC, gradf, grad, boundary, faces)
    end
end