export grad!
export source!

function source!(grad::Grad{S,I,F}, phif, phi, BCs; source=true) where {S,I,F}
    grad!(grad, phif, phi, BCs; source=source)
end

# Mid-point gradient calculation

function grad!(grad::Grad{Midpoint,TI,TF}, phif, phi, BCs; source=false) where {TI,TF}
    interpolate!(get_scheme(grad), phif, phi)
    correct_boundaries!(phif, phi, BCs)
    green_gauss!(grad, phif; source)
    for i ∈ 1:2
        correct_interpolation!(grad, phif, phi)
        green_gauss!(grad, phif; source)
    end
end

function correct_interpolation!(
    grad::Grad{Midpoint,TI,TF}, phif::FaceScalarField{TI,TF}, phi::ScalarField{TI,TF}
    ) where {TI,TF}
    (; mesh, values) = phif
    (; faces, cells) = mesh
    phic = phi.values
    nbfaces = total_boundary_faces(mesh)
    start = nbfaces + 1
    @inbounds @simd for fID ∈ start:length(faces)
        face = faces[fID]
        ownerCells = face.ownerCells
        owner1 = ownerCells[1]
        owner2 = ownerCells[2]
        cell1 = cells[owner1]
        cell2 = cells[owner2]
        phi1 = phic[owner1]
        phi2 = phic[owner2]
        ∇phi1 = grad(owner1)
        ∇phi2 = grad(owner2)
        weight = 0.5
        rf = face.centre 
        rP = cell1.centre 
        rN = cell2.centre
        phifᵖ = weight*(phi1 + phi2)
        ∇phi = weight*(∇phi1 + ∇phi2)
        R = rf - weight*(rP + rN)
        values[fID] = phifᵖ + ∇phi⋅R
    end
end

# Linear gradient calculation

function grad!(grad::Grad{Linear,TI,TF}, phif, phi, BCs; source=false) where {TI,TF}
    # interpolate!(get_scheme(grad), phif, phi) # Needs to be implemented properly
    interpolate!(phif, phi)
    correct_boundaries!(phif, phi, BCs)
    green_gauss!(grad, phif; source)
    # for i ∈ 1:2
    #     correct_interpolation!(grad, phif, phi)
    #     green_gauss!(grad, phif; source)
    # end
end