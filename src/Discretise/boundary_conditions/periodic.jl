export Periodic
export construct_periodic
export adjust_boundary!

struct Periodic{I,V} <: AbstractBoundary
    ID::I
    value::V
end
Adapt.@adapt_structure Periodic

function fixedValue(BC::Periodic, ID::I, value::V) where {I<:Integer,V}
    # Exception 1: Value is scalar
    if V <: Number
        return Periodic{I,typeof(value)}(ID, value)
        # Exception 2: value is a tupple
    elseif V <: NamedTuple
        return Periodic{I,V}(ID, value)
    # Error if value is not scalar or tuple
    else
        throw("The value provided should be a scalar or a tuple")
    end
end

function construct_periodic(
    mesh, backend, patch1::Symbol, patch2::Symbol
    )

    # backend = _get_backend(mesh)
    (; faces, boundaries) = mesh

    boundary_information = boundary_map(mesh)
    idx1 = boundary_index(boundary_information, patch1)
    idx2 = boundary_index(boundary_information, patch2)

    face1 = faces[boundaries[idx1].IDs_range[1]]
    face2 = faces[boundaries[idx2].IDs_range[1]]
    distance = abs((face1.centre - face2.centre)⋅face1.normal)

    nfaces = length(boundaries[idx1].IDs_range)
    faceAddress1 = zeros(Int64, nfaces)
    faceAddress2 = zeros(Int64, nfaces)
    testData = zeros(Float64, nfaces)

    faceCentres1 = getproperty.(faces[boundaries[idx1].IDs_range], :centre)
    faceCentres2 = getproperty.(faces[boundaries[idx2].IDs_range], :centre)

    patchTranslated1 = faceCentres1 .- [distance*face1.normal]
    for (i, face) ∈ enumerate(faces[boundaries[idx2].IDs_range])
        testData .= norm.(patchTranslated1 .- [face.centre])
        val, idx = findmin(testData)
        faceAddress1[i] = boundaries[idx2].IDs_range[idx]
    end

    patchTranslated2 = faceCentres2 .- [distance*face2.normal]
    for (i, face) ∈ enumerate(faces[boundaries[idx1].IDs_range])
        testData .= norm.(patchTranslated2 .- [face.centre])
        val, idx = findmin(testData)
        faceAddress2[i] = boundaries[idx1].IDs_range[idx]
    end

    values1 = (distance=distance, face_map=faceAddress1)
    values2 = (distance=distance, face_map=faceAddress2)
    # periodic1 = Periodic(patch1, values1)
    # periodic2 = Periodic(patch2, values2)
    periodic1 = adapt(backend, Periodic(patch1, values1))
    periodic2 = adapt(backend, Periodic(patch2, values2))
    
    return periodic1, periodic2
end

# Periodic boundary condition assignment

@inline (bc::Periodic)(
    term::Operator{F,P,I,Laplacian{Linear}}, cellID, zcellID, cell, face, fID, i, component=nothing
    ) where {F,P,I} = begin
    phi = term.phi
    mesh = phi.mesh 
    (; faces, cells) = mesh
    values = get_values(phi, component)

    # determine id of periodic cell and interpolate face value
    pfID = bc.value.face_map[i] # id of periodic face 
    pface = faces[pfID]
    pcellID = pface.ownerCells[1]
    pcell = cells[pcellID]

    # face_value = 0.5*(values[cellID] + values[pcellID])
    
    # Retrieve mesh centre values
    xf = face.centre
    xC = cell.centre
    xN = pcell.centre

    delta1 = face.delta
    delta2 = pface.delta
    delta = delta1 + delta2
    # delta = face.delta
    
    # Calculate weights using normal functions
    weight = norm(xf - xC)/(norm(xN - xC) - bc.value.distance)
    # weight = norm(xf - xC)/delta
    # weight = delta1/delta
    one_minus_weight = one(eltype(weight)) - weight

    face_value = weight*values[cellID] + one_minus_weight*values[pcellID]

    # Retrieve term flux and extract fields from workitem face
    J = term.flux[fID]
    # (; area, delta) = face 
    # delta *= 2 # assumes that the distance for the matching patch is the same for now!
    (; area) = face 

    # Calculate flux and ap value for increment
    flux = J*area/delta
    ap = term.sign[1]*(-flux)
    
    ap, ap*face_value
end

@inline (bc::Periodic)(
    term::Operator{F,P,I,Divergence{Linear}}, cellID, zcellID, cell, face, fID, i, component=nothing) where {F,P,I} = begin

    phi = term.phi
    mesh = phi.mesh 
    (; faces, cells) = mesh
    values = get_values(phi, component)

    # determine id of periodic cell and interpolate face value
    pfID = bc.value.face_map[i] # id of periodic face 
    pface = faces[pfID]
    pcellID = pface.ownerCells[1]
    pcell = cells[pcellID]

    # Retrieve mesh centre values
    xf = face.centre
    xC = cell.centre
    xN = pcell.centre

    # delta1 = face.delta
    # delta2 = pface.delta
    # delta = delta1 + delta2
    
    # Calculate weights using normal functions
    weight = norm(xf - xC)/(norm(xN - xC) - bc.value.distance)
    # weight = delta1/delta
    one_minus_weight = one(eltype(weight)) - weight

    # face_value = 0.5*(values[cellID] + values[pcellID])
    face_value = weight*values[cellID] + one_minus_weight*values[pcellID]

    # Calculate ap value to increment
    ap = term.sign[1]*(term.flux[fID])

    # Increment b array
    # Atomix.@atomic b[cellID] -= ap*bc.value
    # nothing
    0.0, -ap*face_value
end

@inline (bc::Periodic)(
    term::Operator{F,P,I,Divergence{Upwind}}, cellID, zcellID, cell, face, fID, i, component=nothing) where {F,P,I} = begin

    phi = term.phi
    mesh = phi.mesh 
    (; faces) = mesh
    values = get_values(phi, component)

    # determine id of periodic cell and interpolate face value
    pfID = bc.value.face_map[i] # id of periodic face 
    pface = faces[pfID]
    pcellID = pface.ownerCells[1]
    face_value = 0.5*(values[cellID] + values[pcellID])

    # Calculate ap value to increment
    ap = term.sign[1]*(term.flux[fID])

    # Increment b array
    # Atomix.@atomic b[cellID] -= ap*bc.value
    # nothing
    0.0, -ap*face_value
end

# # Boundary interpolation

# function adjust_boundary!(b_cpu, BC::Periodic, phif::FaceScalarField, phi, boundaries, boundary_cellsID,  backend, workgroup)
#     phif_values = phif.values
#     phi_values = phi.values

#     # Copy to CPU
#     # facesID_range = get_boundaries(BC, boundaries)
#     kernel_range = length(b_cpu[BC.ID].IDs_range)

#     kernel! = adjust_boundary_dirichlet_scalar!(backend, workgroup)
#     kernel!(BC, phif, phi, boundaries, boundary_cellsID, phif_values, phi_values, ndrange = kernel_range)
#     # KernelAbstractions.synchronize(backend)
# end

# @kernel function adjust_boundary_dirichlet_scalar!(BC, phif, phi, boundaries, boundary_cellsID, phif_values, phi_values)
#     i = @index(Global)
#     # i = BC.ID

#     @inbounds begin
#         # (; IDs_range) = boundaries[BC.ID]
#         (; IDs_range) = boundaries[BC.ID]
#         fID = IDs_range[i]
#         # for fID in IDs_range
#             phif_values[fID] = BC.value
#         # end
#     end
# end

# function adjust_boundary!(b_cpu, BC::Periodic, psif::FaceVectorField, psi::VectorField, boundaries, boundary_cellsID, backend, workgroup)
#     (; x, y, z) = psif

#     kernel_range = length(b_cpu[BC.ID].IDs_range)

#     kernel! = adjust_boundary_dirichlet_vector!(backend, workgroup)
#     kernel!(BC, psif, psi, boundaries, boundary_cellsID, x, y, z, ndrange = kernel_range)
#     # KernelAbstractions.synchronize(backend)
# end

# @kernel function adjust_boundary_dirichlet_vector!(BC, psif, psi, boundaries, boundary_cellsID, x, y, z)
#     i = @index(Global)
#     # i = BC.ID

#     @inbounds begin
#         # (; IDs_range) = boundaries[i]
#         (; IDs_range) = boundaries[BC.ID]
#         # for fID in IDs_range
#         fID = IDs_range[i]
#             # fID = IDs_range[j]
#             x[fID] = BC.value[1]
#             y[fID] = BC.value[2]
#             z[fID] = BC.value[3]
#         # end
#     end
# end