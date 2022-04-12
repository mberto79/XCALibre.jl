export interpolate!

# Scalar face interpolation

function interpolate!(::Type{Linear}, phif::FaceScalarField{I,F}, phi) where {I,F}
    nbfaces = total_boundary_faces(phi.mesh)
    start = nbfaces + 1
    (; mesh, values) = phi
    (; cells, faces) = mesh
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
    for i ∈ 1:nbfaces
        (; ownerCells) = faces[i]
        c1 = ownerCells[1]
        phif.values[i] = values[c1]
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

# Gradient (vector?) face interpolation

function interpolate!(::Type{Linear}, gradf, grad)
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
    for i ∈ 1:nbfaces
        (; ownerCells) = faces[i]
        c1 = ownerCells[1]
        grad1 = grad(c1)
        x[i] = grad1[1]
        y[i] = grad1[2]
        z[i] = grad1[3]
    end
end

function correct_interpolation!(::Type{Linear}, gradf, grad)
    mesh = phif.mesh
    start = total_boundary_faces(mesh) + 1
    (; cells, faces) = mesh
    for fi ∈ start:length(faces)
        (; ownerCells, centre, area, normal) = faces[fi]
        w, df = weight(Linear, cells, faces, fi)
        cID1 = ownerCells[1]
        cID2 = ownerCells[2]
        c1 = cells[cID1].centre
        c2 = cells[cID2].centre
        c1_f = centre - c1
        c2_f = centre - c2
        grad1 = grad(cID1)
        grad2 = grad(cID2)
        grad_ave = w*grad1 + (1.0 - w)*grad2
        phif.values[fi] += grad_ave⋅df
    end
end

# Weight functions

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