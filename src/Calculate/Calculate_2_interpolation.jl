export correct_boundaries!
export interpolate!

# Function to correct interpolation at boundaries (expands loop to reduce allocations)

@generated function correct_boundaries!(phif, phi, BCs)
    unpacked_BCs = []
    for i ∈ 1:length(BCs.parameters)
        unpack = quote
            BC = BCs[$i]
            # name = BC.name
            # index = boundary_index(boundaries, name)
            boundary = boundaries[BC.ID]
            adjust_boundary!(BC, phif, phi, boundary, faces)
            # adjust_boundary!(BC, phif, phi, BC.ID, faces)
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
    BC::Dirichlet, phif::FaceScalarField, phi, boundary, faces)
    # (;facesID, cellsID) = boundary
    (; IDs_range) = boundary
    @inbounds for fID ∈ IDs_range
        phif.values[fID] = BC.value 
    end
end

function adjust_boundary!(
    BC::Neumann, phif::FaceScalarField, phi, boundary, faces)
    # (;facesID, cellsID) = boundary
    (; IDs_range) = boundary
    (; boundary_cellsID) = phif.mesh
    @inbounds for fi ∈ IDs_range
        # fID = facesID[fi]
        fID = fi
        cID = boundary_cellsID[fi]
        # (; normal, e, delta) = faces[fID]
        # phif.values[fID] = phi.values[cID] #+ BC.value*delta*(normal⋅e)
        # Chris' fix
        (; area, normal, e, delta) = faces[fID]
        phif.values[fID] = phi.values[cID] + BC.value*delta
    end
end

function adjust_boundary!(
    BC::Wall, phif::FaceScalarField, phi, boundary, faces)
    (; facesID, cellsID) = boundary
    @inbounds for fID ∈ facesID
        phif.values[fID] = BC.value 
    end
end

function adjust_boundary!(
    BC::Symmetry, phif::FaceScalarField, phi, boundary, faces)
    (; facesID, cellsID) = boundary
    @inbounds for fID ∈ facesID
        phif.values[fID] = psi.value 
    end
end

function adjust_boundary!(
    BC::KWallFunction, phif::FaceScalarField, phi, boundary, faces)
    # (;facesID, cellsID) = boundary
    (; IDs_range) = boundary
    (; boundary_cellsID) = phif.mesh
    @inbounds for fi ∈ IDs_range
        # fID = facesID[fi]
        # cID = cellsID[fi]
        fID = fi
        cID = boundary_cellsID[fID]
        phif.values[fID] = phi.values[cID] # Using Neumann condition
    end
end

function adjust_boundary!(
    BC::NutWallFunction, phif::FaceScalarField, phi, boundary, faces)
    # (;facesID, cellsID) = boundary
    (; IDs_range) = boundary
    (; boundary_cellsID) = phif.mesh
    @inbounds for fi ∈ IDs_range
        # fID = facesID[fi]
        # cID = cellsID[fi]
        fID = fi
        cID = boundary_cellsID[fID]
        phif.values[fID] = phi.values[cID] # Using Neumann condition
    end
end

function adjust_boundary!(
    BC::OmegaWallFunction, phif::FaceScalarField, phi, boundary, faces)
    # (;facesID, cellsID) = boundary
    (; IDs_range) = boundary
    (; boundary_cellsID) = phif.mesh
    @inbounds for fi ∈ IDs_range
        # fID = facesID[fi]
        # cID = cellsID[fi]
        fID = fi
        cID = boundary_cellsID[fID]
        phif.values[fID] = phi.values[cID] # Using Neumann condition
    end
end

function adjust_boundary!( 
    BC::Dirichlet, psif::FaceVectorField, psi::VectorField, boundary, faces
    )

    (; x, y, z) = psif
    (; IDs_range) = boundary

    @inbounds for fID ∈ IDs_range
        x[fID] = BC.value[1]
        y[fID] = BC.value[2]
        z[fID] = BC.value[3]
    end
end

function adjust_boundary!( 
    BC::Neumann, psif::FaceVectorField, psi::VectorField, boundary, faces
    ) 

    (; x, y, z, mesh) = psif
    (; boundary_cellsID) = mesh
    (; IDs_range) = boundary

    @inbounds for fi ∈ IDs_range
        fID = fi
        cID = boundary_cellsID[fID]
        psi_cell = psi[cID]
        # normal = faces[fID].normal
        # Line below needs sorting out for general user-defined gradients
        # now only works for zero gradient
        # psi_boundary =   psi_cell - (psi_cell⋅normal)*normal
        x[fID] = psi_cell[1]
        y[fID] = psi_cell[2]
        z[fID] = psi_cell[3]
    end
end

function adjust_boundary!( 
    BC::Wall, psif::FaceVectorField, psi::VectorField, boundary, faces
    )

    (; x, y, z) = psif
    (; IDs_range) = boundary

    @inbounds for fID ∈ IDs_range
        x[fID] = BC.value[1]
        y[fID] = BC.value[2]
        z[fID] = BC.value[3]
    end
end

function adjust_boundary!( 
    BC::Symmetry, psif::FaceVectorField, psi::VectorField, boundary, faces
    )

    (; x, y, z) = psif
    (; IDs_range) = boundary
    (; boundary_cellsID) = psif.mesh

    @inbounds for fID ∈ IDs_range
        (; area, normal, e, delta) = faces[fID]
        cID = boundary_cellsID[fID]
        x[fID] = psi.x.values[cID] - (psi.x.values[cID]*normal[1] + psi.y.values[cID]*normal[2] + psi.z.values[cID]*normal[3])*normal[1]
        y[fID] = psi.y.values[cID] - (psi.x.values[cID]*normal[1] + psi.y.values[cID]*normal[2] + psi.z.values[cID]*normal[3])*normal[2]
        z[fID] = psi.z.values[cID] - (psi.x.values[cID]*normal[1] + psi.y.values[cID]*normal[2] + psi.z.values[cID]*normal[3])*normal[3]
    end
end

# SCALAR INTERPOLATION

function interpolate!(phif::FaceScalarField, phi::ScalarField) 
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
        one_minus_weight = 1 - weight
        fvals[fID] = weight*phi1 + one_minus_weight*phi2 # check weight is used correctly!
    end
end

# VECTOR INTERPOLATION

function interpolate!(psif::FaceVectorField, psi::VectorField)
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
        one_minus_weight = 1 - weight
        x[fID] = weight*x1 + one_minus_weight*x2 # check weight is used correctly!
        y[fID] = weight*y1 + one_minus_weight*y2 # check weight is used correctly!
    end
end

# GRADIENT INTERPOLATION

function interpolate!(
    gradf::FaceVectorField, grad::Grad, phi
    )
    (; mesh, x, y, z) = gradf
    (; cells, faces) = mesh
    (; values) = phi
    nbfaces = total_boundary_faces(mesh)
    start = nbfaces + 1
    @inbounds for fID ∈ start:length(faces)
        face = faces[fID]
        (; delta, ownerCells, e) = face
        cID1 = ownerCells[1]
        cID2 = ownerCells[2]
        grad1 = grad(cID1)
        grad2 = grad(cID2)
        # get weight for current scheme
        w, df = weight(get_scheme(grad), cells, faces, fID)
        one_minus_weight = 1 - w
        # calculate interpolated value
        grad_ave = w*grad1 + one_minus_weight*grad2
        # correct interpolation
        grad_corr = grad_ave + ((values[cID2] - values[cID1])/delta - (grad_ave⋅e))*e
        x[fID] = grad_corr[1]
        y[fID] = grad_corr[2]
        z[fID] = grad_corr[3]
    end
end

# function weight(::Type{Midpoint}, cells, faces, fID)
#     w = 0.5
#     return w
# end

# function correct_gradient_interpolation!(::Type{Linear}, gradf, phi)
#     values = phi.phi.values
#     mesh = phi.mesh
#     (; cells, faces) = mesh
#     start = total_boundary_faces(mesh) + 1
#     finish = length(faces)
#     @inbounds for fID ∈ start:finish
#     # for fi ∈ 1:length(faces)
#         # (; ownerCells, delta, e) = faces[fi]
#         face = faces[fID]
#         ownerCells = face.ownerCells
#         delta = face.ownerCells
#         e = face.e
#         w, df = weight(Linear, cells, faces, fi)
#         cID1 = ownerCells[1]
#         cID2 = ownerCells[2]
#         grad_ave = gradf(fID)
#         grad_corr = grad_ave + ((values[cID2] - values[cID1])/delta - (grad_ave⋅e))*e
#         gradf.x[fID] = grad_corr[1]
#         gradf.y[fID] = grad_corr[2]
#         gradf.z[fID] = grad_corr[3]
#     end
# end

############ OLD LINEAR GRADIENT INTERPOLATION IMPLEMENTATION #############


# function interpolate!(::Type{Linear}, gradf::FaceVectorField, grad, BCs)
#     (; mesh, x, y, z) = gradf
#     (; cells, faces) = mesh
#     nbfaces = total_boundary_faces(mesh)
#     start = nbfaces + 1
#     @inbounds for fID ∈ start:length(faces)
#         # (; ownerCells) = faces[fi]
#         face = faces[fID]
#         ownerCells = face.ownerCells
#         w, df = weight(Linear, cells, faces, fID)
#         cID1 = ownerCells[1]
#         cID2 = ownerCells[2]
#         # grad1 = grad(cID1)
#         # grad2 = grad(cID2)
#         grad1 = grad(cID1)
#         grad2 = grad(cID2)
#         one_minus_weight = 1.0 - w
#         gradi = w*grad1 + one_minus_weight*grad2
#         x[fID] = gradi[1]
#         y[fID] = gradi[2]
#         z[fID] = gradi[3]
#     end
#     correct_gradient_interpolation!(Linear, gradf, grad)
#     # boundary faces
#     for BC ∈ BCs
#         bi = boundary_index(boundaries, BC.name)
#         boundary = boundaries[bi]
#         correct_boundary!(BC, gradf, grad, boundary, faces)
#     end
# end

# function correct_interpolation!(
#     ::Type{Linear}, phif::FaceScalarField{I,F}, grad, phif0) where {I,F}
#     mesh = phif.mesh
#     (; cells, faces) = mesh
#     start = total_boundary_faces(mesh) + 1
#     finish = length(faces)
#     @inbounds for fID ∈ start:finish
#         # (; ownerCells) = faces[fi]
#         face = faces[fID]
#         ownerCells = face.ownerCells
#         w, df = weight(Linear, cells, faces, fi)
#         cID1 = ownerCells[1]
#         cID2 = ownerCells[2]
#         grad1 = grad(cID1)
#         grad2 = grad(cID2)
#         one_minus_weight = 1.0 - w
#         grad_ave = w*grad1 + one_minus_weight*grad2
#         phif.values[fID] = phif0[fID] + grad_ave⋅df
#     end
# end

# function correct_gradient_interpolation!(::Type{Linear}, gradf, phi)
#     values = phi.phi.values
#     mesh = phi.mesh
#     (; cells, faces) = mesh
#     start = total_boundary_faces(mesh) + 1
#     finish = length(faces)
#     @inbounds for fID ∈ start:finish
#     # for fi ∈ 1:length(faces)
#         # (; ownerCells, delta, e) = faces[fi]
#         face = faces[fID]
#         ownerCells = face.ownerCells
#         delta = face.ownerCells
#         e = face.ownerCells
#         w, df = weight(Linear, cells, faces, fi)
#         cID1 = ownerCells[1]
#         cID2 = ownerCells[2]
#         # c1 = cells[cID1].centre
#         # c2 = cells[cID2].centre
#         # distance = c2 - c1
#         # d = distance/delta
#         d = e
#         # grad1 = grad(cID1)
#         # grad2 = grad(cID2)
#         # grad_ave = w*grad1 + (1.0 - w)*grad2
#         grad_ave = gradf(fID)
#         grad_corr = grad_ave + ((values[cID2] - values[cID1])/delta - (grad_ave⋅d))*d
#         gradf.x[fID] = grad_corr[1]
#         gradf.y[fID] = grad_corr[2]
#         gradf.z[fID] = grad_corr[3]
#     end
# end

# function correct_boundary!( # Another way is to use the boundary value and geometry to calc
#     BC::Dirichlet, gradf::FaceVectorField, grad, boundary, faces)
#     (; mesh, x, y, z) = gradf
#     (; facesID) = boundary
#     @inbounds for fID ∈ facesID
#         face = faces[fID]
#         normal = faces[fID].normal
#         cID = face.ownerCells[1]
#         grad_cell = grad(cID)
#         # grad_cell = grad[cID]
#         grad_boundary = grad_cell #.*normal .+ grad_cell
#         # grad_boundary = ((BC.value - grad.phi.values[fID])/face.delta)*normal
#         x[fID] = grad_boundary[1]
#         y[fID] = grad_boundary[2]
#         z[fID] = grad_boundary[3]
#     end
# end

# function correct_boundary!(
#     BC::Neumann, gradf::FaceVectorField, grad, boundary, faces)
#     (; mesh, x, y, z) = gradf
#     (; facesID) = boundary
#     @inbounds for fID ∈ facesID
#         face = faces[fID]
#         normal = faces[fID].normal
#         cID = face.ownerCells[1]
#         grad_cell = grad(cID)
#         # grad_cell = grad[cID]
#         grad_boundary =   grad_cell - (grad_cell⋅normal)*normal # needs sorting out!
#         x[fID] = grad_boundary[1]
#         y[fID] = grad_boundary[2]
#         z[fID] = grad_boundary[3]
#     end
# end