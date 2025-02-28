export Periodic, PeriodicConnectivity
export construct_periodic
export adjust_boundary!


"""
    struct Periodic{I,V} <: AbstractPhysicalConstraint
        ID::I
        value::V
    end

Periodic boundary condition model.

### Fields
- 'ID' -- Boundary ID
- `value` -- tuple containing information needed to apply this boundary
"""
struct Periodic{I,V} <: AbstractPhysicalConstraint
    ID::I
    value::V
end
Adapt.@adapt_structure Periodic

struct PeriodicConnectivity{I}
    i::I
    j::I
end
Adapt.@adapt_structure PeriodicConnectivity

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

"""
    construct_periodic(mesh, backend, patch1::Symbol, patch2::Symbol)

Function for construction of periodic boundary conditions.

### Input
- `mesh` -- Mesh.
- `backend`  -- Backend configuraton.
- `patch1`  -- Primary periodic patch ID.
- `patch2`   -- Neighbour periodic patch ID.

### Output
- periodic::Tuple - tuple containing boundary defintions for `patch1` and `patch2` i.e. (periodic1, periodic2). The fields of `periodic1` and `periodic2` are 

    - `ID` -- Index to access boundary information in mesh object
    - `value` -- represents a `NamedTuple` with the following keyword arguments:
        - index -- ID used to find boundary geometry information in the mesh object
        - distance -- perpendicular distance between the patches
        - face_map -- vector providing indeces to faces of match patch
        - ismaster -- flat to identify one of the patch pairs as the main patch

### Example
    - `periodic = construct_periodic(mesh, CPU(), :top, :bottom)` - Example using CPU 
    backend with periodic boundaries named `top` and `bottom`.

"""
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

function periodic_matrix_connectivity(mesh, patch1, patch2)
    (; faces, cells, boundaries, boundary_cellsID) = mesh
    boundary_information = boundary_map(mesh)
    idx1 = boundary_index(boundary_information, patch1.ID)
    idx2 = boundary_index(boundary_information, patch2.ID)

    BC1 = boundaries[idx1].IDs_range
    BC2 = boundaries[idx2].IDs_range

    fmap1 = patch1.value.face_map
    fmap2 = patch2.value.face_map
    # i = zeros(Int, length(fmap2))
    # j = zeros(Int, length(fmap2))
    i = zeros(Int, 2*length(fmap2))
    j = zeros(Int, 2*length(fmap2))
    nindex = 0

    for (fID1, fID2) ∈ zip(BC1, fmap1) # swap order to get correct fID
        face1 = faces[fID1]
        face2 = faces[fID2]
        cID1 = face1.ownerCells[1]
        cID2 = face2.ownerCells[1]

        nindex += 1
        i[nindex] = cID1
        j[nindex] = cID2

        nindex += 1
        i[nindex] = cID2
        j[nindex] = cID1
    end

    PeriodicConnectivity(i , j)
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
    ap = term.sign*(term.flux[fID]*area)/delta
    ac = -ap
    an = ap

    # Playing with implicit version
    # fzcellID = spindex(rowptr, colval, cellID, pcellID)
    # # fzcellID = spindex(rowptr, colval, pcellID, cellID)
    # Atomix.@atomic nzval[fzcellID] = an
    # ac, 0.0

    ac, -an*values[pcellID] # explicit this works
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
    
    weight = delta2/delta
    one_minus_weight = one(eltype(weight)) - weight

    # Calculate ap value to increment
    flux = term.flux[fID]
    ap = term.sign*(flux)
    ac = weight*ap
    an = one_minus_weight*ap

    # Playing with implicit version
    fzcellID = spindex(rowptr, colval, cellID, pcellID)
    # fzcellID = spindex(rowptr, colval, pcellID, cellID)
    Atomix.@atomic nzval[fzcellID] = an
    ac, 0.0

    # ac, -an*values[pcellID] # explicit this works
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

@define_boundary Periodic Divergence{LUST} begin
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