export Periodic, PeriodicConnectivity
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
    for (i, face) ∈ enumerate(faces[boundaries[idx2].IDs_range]) # use @view?
        testData .= norm.(patchTranslated1 .- [face.centre])
        val, idx = findmin(testData)
        faceAddress1[i] = boundaries[idx2].IDs_range[idx]
    end

    patchTranslated2 = faceCentres2 .- [distance*face2.normal] # use @view?
    for (i, face) ∈ enumerate(faces[boundaries[idx1].IDs_range])
        testData .= norm.(patchTranslated2 .- [face.centre])
        val, idx = findmin(testData)
        faceAddress2[i] = boundaries[idx1].IDs_range[idx]
    end

    values1 = (index=idx1, distance=distance, face_map=faceAddress1, ismaster=true)
    values2 = (index=idx2, distance=distance, face_map=faceAddress2, ismaster=false)

    p1 = Periodic(patch1, values1)
    p2 = Periodic(patch2, values2)

    periodic1 = adapt(backend, p1)
    periodic2 = adapt(backend, p2)
    
    return (periodic1, periodic2)
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

    # for improved accuracy this needs to include the discretisation used for noncorrection
    delta1 = face.delta
    delta2 = pface.delta
    delta = delta1 + delta2
    
    # Retrieve term flux and extract fields from workitem face
    (; area, normal) = face
    J = term.flux[fID]
    flux = J*area/delta
    ap = term.sign*(-flux)

    # Explicit allowing looping over slave patch
    ap, ap*values[pcellID] # explicit this works

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


    delta1 = face.delta
    delta2 = pface.delta
    delta = delta1 + delta2
    
    weight = delta1/delta
    one_minus_weight = one(eltype(weight)) - weight

    # Calculate ap value to increment
    flux = term.flux[fID]
    ap = term.sign*(flux)
    ac = weight*ap
    an = one_minus_weight*ap

    # Explicit allowing looping over slave patch
    ac, -an*values[pcellID] # explicit this works
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


    delta1 = face.delta
    delta2 = pface.delta
    delta = delta1 + delta2

    weight = delta1/delta
    one_minus_weight = one(eltype(weight)) - weight

    # Calculate ap value to increment
    flux = term.flux[fID]
    ap = term.sign*(flux)
    ac = weight*ap
    an = one_minus_weight*ap

    # Explicit allowing looping over slave patch
    ac, -an*values[pcellID] # explicit this works
end