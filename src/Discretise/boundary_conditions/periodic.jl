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

function construct_periodic(mesh, backend, patch1::Symbol, patch2::Symbol)

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
    periodic1 = adapt(backend, Periodic(patch1, values1))
    periodic2 = adapt(backend, Periodic(patch2, values2))
    
    return periodic1, periodic2
end


@define_boundary Periodic Laplacian{Linear} begin
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
    # xf = face.centre
    # xC = cell.centre
    # xN = pcell.centre

    delta1 = face.delta
    delta2 = pface.delta
    delta = delta1 + delta2
    # delta = face.delta
    
    # Calculate weights using normal functions
    # weight = norm(xf - xC)/(norm(xN - xC) - bc.value.distance)
    # weight = 0.5
    # weight = norm(xf - xC)/delta
    weight = delta1/delta
    one_minus_weight = one(eltype(weight)) - weight

    face_value = weight*values[cellID] + one_minus_weight*values[pcellID]

    # Retrieve term flux and extract fields from workitem face
    J = term.flux[fID]
    # (; area, delta) = face 
    # delta *= 2 # assumes that the distance for the matching patch is the same for now!
    (; area) = face 

    # Calculate flux and ap value for increment
    # flux = J*area/delta # original
    flux = J*area/delta1
    ap = term.sign[1]*(-flux)
    
    ap, ap*face_value # original
    # weight*ap, ap*one_minus_weight*values[pcellID]
end

@define_boundary Periodic Divergence{Linear} begin
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
    # xf = face.centre
    # xC = cell.centre
    # xN = pcell.centre

    delta1 = face.delta
    delta2 = pface.delta
    delta = delta1 + delta2
    
    # Calculate weights using normal functions
    # weight = norm(xf - xC)/(norm(xN - xC) - bc.value.distance)
    weight = delta1/delta
    one_minus_weight = one(eltype(weight)) - weight

    # face_value = 0.5*(values[cellID] + values[pcellID])
    face_value = weight*values[cellID] + one_minus_weight*values[pcellID]

    # Calculate ap value to increment
    ap = term.sign[1]*(term.flux[fID])

    0.0, -ap*face_value
    # weight*ap, -ap*one_minus_weight*values[pcellID]
end

@define_boundary Periodic Divergence{Upwind} begin
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
    # xf = face.centre
    # xC = cell.centre
    # xN = pcell.centre

    delta1 = face.delta
    delta2 = pface.delta
    delta = delta1 + delta2

    # Calculate weights using normal functions
    # weight = norm(xf - xC)/(norm(xN - xC) - bc.value.distance)
    weight = delta1/delta
    one_minus_weight = one(eltype(weight)) - weight

    # face_value = 0.5*(values[cellID] + values[pcellID])
    face_value = weight*values[cellID] + one_minus_weight*values[pcellID]

    # Calculate ap value to increment
    ap = term.sign[1]*(term.flux[fID])

    0.0, -ap*face_value
    # weight*ap, -ap*one_minus_weight*values[pcellID]
end