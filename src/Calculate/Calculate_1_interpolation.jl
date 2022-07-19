export interpolate!
export correct_boundaries!

# Function to correct interpolation at boundaries (expands loop to reduce allocations)

@generated function correct_boundaries!(phif, phi, BCs)
    unpacked_BCs = []
    for i ∈ 1:length(BCs.parameters)
        unpack = quote
            BC = BCs[$i]
            name = BC.name
            index = boundary_index(boundaries, name)
            boundary = boundaries[index]
            adjust_boundary!(BC, phif, phi, boundary, faces)
        end
        push!(unpacked_BCs, unpack)
    end
    quote
    mesh = phi.mesh
    (; faces, boundaries) = mesh  
    $(unpacked_BCs...) 
    end
end

function adjust_boundary!(
    BC::Dirichlet, phif::FaceScalarField{I,F}, phi, boundary, faces) where {I,F}
    (; facesID, cellsID) = boundary
    @inbounds for fID ∈ facesID
        phif.values[fID] = BC.value 
    end
end

function adjust_boundary!(
    BC::Neumann, phif::FaceScalarField{I,F}, phi, boundary, faces) where {I,F}
    (;facesID, cellsID) = boundary
    @inbounds for fi ∈ eachindex(facesID)
        fID = facesID[fi]
        cID = cellsID[fi]
        # (; normal, e, delta) = faces[fID]
        phif.values[fID] = phi.values[cID] #+ BC.value*delta*(normal⋅e)
    end
end

function adjust_boundary!( 
    BC::Dirichlet, psif::FaceVectorField{I,F}, psi::VectorField{I,F}, boundary, faces
    ) where {I,F}

    (; x, y, z) = psif
    (; facesID) = boundary

    @inbounds for fID ∈ facesID
        x[fID] = BC.value[1]
        y[fID] = BC.value[2]
        z[fID] = BC.value[3]
    end
end

function adjust_boundary!( 
    BC::Neumann, psif::FaceVectorField{I,F}, psi::VectorField{I,F}, boundary, faces
    ) where {I,F}

    (; x, y, z) = psif
    (; facesID, cellsID) = boundary

    @inbounds for fi ∈ eachindex(facesID)
        fID = facesID[fi]
        cID = cellsID[fi]
        psi_cell = psi(cID)
        # normal = faces[fID].normal
        # Line below needs sorting out for general user-defined gradients
        # now only works for zero gradient
        # psi_boundary =   psi_cell - (psi_cell⋅normal)*normal
        x[fID] = psi_cell[1]
        y[fID] = psi_cell[2]
        z[fID] = psi_cell[3]
    end
end

# SCALAR INTERPOLATION

function interpolate!(phif::FaceScalarField{I,F}, phi::ScalarField{I,F}) where {I,F}
    vals = phi.values 
    fvals = phif.values
    mesh = phi.mesh 
    faces = mesh.faces
    @inbounds for fID ∈ eachindex(faces)
        # (; weight, ownerCells) = faces[fi]
        face = faces[fID]
        weight = face.weight
        ownerCells = face.ownerCells
        phi1 = vals[ownerCells[1]]
        phi2 = vals[ownerCells[2]]
        one_minus_weight = 1.0 - weight
        fvals[fID] = weight*phi1 + one_minus_weight*phi2 # check weight is used correctly!
    end
end

function interpolate!(
    ::Type{Midpoint}, phif::FaceScalarField{I,F}, phi::ScalarField{I,F}
    ) where {I,F}
    vals = phi.values 
    fvals = phif.values
    mesh = phi.mesh 
    faces = mesh.faces
    nbfaces = total_boundary_faces(mesh)
    start = nbfaces + 1
    @inbounds for fID ∈ start:length(faces)
        face = faces[fID]
        weight = 0.5
        ownerCells = face.ownerCells
        phi1 = vals[ownerCells[1]]
        phi2 = vals[ownerCells[2]]
        one_minus_weight = 1.0 - weight
        fvals[fID] = weight*phi1 + one_minus_weight*phi2
    end
end

# VECTOR INTERPOLATION

function interpolate!(psif::FaceVectorField{I,F}, psi::VectorField{I,F}) where {I,F}
    (; x, y, z) = psif # must extend to 3D
    mesh = psi.mesh
    faces = mesh.faces
    @inbounds for fID ∈ eachindex(faces)
        # (; weight, ownerCells) = faces[fID]
        face = faces[fID]
        weight = face.weight
        ownerCells = face.ownerCells
        # w, df = weight(Linear, cells, faces, fi)
        cID1 = ownerCells[1]; cID2 = ownerCells[2]
        x1 = psi.x[cID1]; x2 = psi.x[cID2]
        y1 = psi.y[cID1]; y2 = psi.y[cID2]
        one_minus_weight = 1.0 - weight
        x[fID] = weight*x1 + one_minus_weight*x2 # check weight is used correctly!
        y[fID] = weight*y1 + one_minus_weight*y2 # check weight is used correctly!
    end
end

# GRADIENT INTERPOLATION

function interpolate!(::Type{Linear}, gradf::FaceVectorField{I,F}, grad, BCs) where {I,F}
    (; mesh, x, y, z) = gradf
    (; cells, faces) = mesh
    nbfaces = total_boundary_faces(mesh)
    start = nbfaces + 1
    @inbounds for fID ∈ start:length(faces)
        # (; ownerCells) = faces[fi]
        face = faces[fID]
        ownerCells = face.ownerCells
        w, df = weight(Linear, cells, faces, fID)
        cID1 = ownerCells[1]
        cID2 = ownerCells[2]
        grad1 = grad(cID1)
        grad2 = grad(cID2)
        one_minus_weight = 1.0 - w
        gradi = w*grad1 + one_minus_weight*grad2
        x[fID] = gradi[1]
        y[fID] = gradi[2]
        z[fID] = gradi[3]
    end
    correct_interpolation!(Linear, gradf, grad)
    # boundary faces
    for BC ∈ BCs
        bi = boundary_index(boundaries, BC.name)
        boundary = boundaries[bi]
        correct_boundary!(BC, gradf, grad, boundary, faces)
    end
end

function correct_interpolation!(
    ::Type{Linear}, phif::FaceScalarField{I,F}, grad, phif0) where {I,F}
    mesh = phif.mesh
    (; cells, faces) = mesh
    start = total_boundary_faces(mesh) + 1
    finish = length(faces)
    @inbounds for fID ∈ start:finish
        # (; ownerCells) = faces[fi]
        face = faces[fID]
        ownerCells = face.ownerCells
        w, df = weight(Linear, cells, faces, fi)
        cID1 = ownerCells[1]
        cID2 = ownerCells[2]
        grad1 = grad(cID1)
        grad2 = grad(cID2)
        one_minus_weight = 1.0 - w
        grad_ave = w*grad1 + one_minus_weight*grad2
        phif.values[fID] = phif0[fID] + grad_ave⋅df
    end
end

function correct_interpolation!(::Type{Linear}, gradf, grad)
    values = grad.phi.values
    mesh = grad.mesh
    (; cells, faces) = mesh
    start = total_boundary_faces(mesh) + 1
    finish = length(faces)
    @inbounds for fID ∈ start:finish
    # for fi ∈ 1:length(faces)
        # (; ownerCells, delta, e) = faces[fi]
        face = faces[fID]
        ownerCells = face.ownerCells
        delta = face.ownerCells
        e = face.ownerCells
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
        grad_ave = gradf(fID)
        grad_corr = grad_ave + ((values[cID2] - values[cID1])/delta - (grad_ave⋅d))*d
        gradf.x[fID] = grad_corr[1]
        gradf.y[fID] = grad_corr[2]
        gradf.z[fID] = grad_corr[3]
    end
end

function correct_boundary!( # Another way is to use the boundary value and geometry to calc
    BC::Dirichlet, gradf::FaceVectorField{I,F}, grad, boundary, faces) where {I,F}
    (; mesh, x, y, z) = gradf
    (; facesID) = boundary
    @inbounds for fID ∈ facesID
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
    @inbounds for fID ∈ facesID
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