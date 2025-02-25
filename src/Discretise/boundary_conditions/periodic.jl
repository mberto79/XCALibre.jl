export Periodic, PeriodicConnectivity
export construct_periodic
export adjust_boundary!

struct Periodic{I,V} <: AbstractBoundary
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

    values1 = (index=idx1, distance=distance, face_map=faceAddress1, ismaster=true)
    values2 = (index=idx2, distance=distance, face_map=faceAddress2, ismaster=false)

    p1 = Periodic(patch1, values1)
    p2 = Periodic(patch2, values2)
    connectivity = periodic_matrix_connectivity(mesh, p1, p2)

    # flip normals for patch2 
    # IDs_range = mesh.boundaries[idx2].IDs_range

    # for fID ∈ IDs_range
    #     face = faces[fID]  
    #     println("before: ", face.normal)
    #     @reset face.normal *= -1 
    #     faces[fID] = face
    #     println("after: ", face.normal)
    # end


    periodic1 = adapt(backend, p1)
    periodic2 = adapt(backend, p2)
    
    return (periodic1, periodic2), connectivity, mesh
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
    i = zeros(Int, 2*length(fmap2))
    j = zeros(Int, 2*length(fmap2))
    # i = zeros(Int, length(fmap2))
    # j = zeros(Int, length(fmap2))
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

    # for (fID1, fID2) ∈ zip(BC2, fmap2) # swap order to get correct fID
    #     face1 = faces[fID1]
    #     face2 = faces[fID2]
    #     cID1 = face1.ownerCells[1]
    #     cID2 = face2.ownerCells[1]

    #     nindex += 1
    #     i[nindex] = cID1
    #     j[nindex] = cID2
    # end
    
    # for (fID1, fID2) ∈ zip(fmap2, fmap1) # swap order to get correct fID
    #     println(fID1," ", fID2)
    #     face1 = faces[fID1]
    #     face2 = faces[fID2]
    #     cID1 = face1.ownerCells[1]
    #     cID2 = face2.ownerCells[1]

    #     nindex += 1
    #     i[nindex] = cID1
    #     j[nindex] = cID2

    #     nindex += 1
    #     i[nindex] = cID2
    #     j[nindex] = cID1
    # end
    PeriodicConnectivity(i , j)
end


@define_boundary Periodic Laplacian{Linear} begin

    # if !bc.value.ismaster
    #     return 0.0, 0.0
    # end

    phi = term.phi
    mesh = phi.mesh 
    (; faces, cells) = mesh
    values = get_values(phi, component)

    # determine id of periodic cell and interpolate face value
    pfID = bc.value.face_map[i] # id of periodic face 
    pface = faces[pfID]
    pcellID = pface.ownerCells[1]
    pcell = cells[pcellID]

    # fcellID = face.ownerCells[1]
    # fcellID = faces[fID].ownerCells[1]

    # Retrieve mesh centre values
    # xf = face.centre
    # xC = cell.centre
    # xN = pcell.centre

    # Calculate weights using normal functions
    # weight = norm(xf - xC)/(norm(xN - xC) - bc.value.distance)
    # weight = 0.5
    # weight = norm(xf - xC)/delta

    delta1 = face.delta
    delta2 = pface.delta
    delta = delta1 + delta2
    
    # when using interpolated face value
    # weight = delta1/delta
    # one_minus_weight = one(eltype(weight)) - weight
    # face_value = weight*values[cellID] + one_minus_weight*values[pcellID]

    # Retrieve term flux and extract fields from workitem face
    J = term.flux[fID]
    (; area) = face 

    # Calculate flux and ap value for increment
    # flux = J*area/delta1 # when using interpolated face value
    # flux = J*area/delta # when using neighbour cell value
    # ap = term.sign[1]*(flux)

    # # implicit implementation
    (; area, normal, e) = face
    # # Sf = ns*area*normal
    # # e = ns*e
    # Sf = area*normal
    # Ef = ((Sf⋅Sf)/(Sf⋅e))*e
    # Ef_mag = norm(Ef)
    # # ap = term.sign[1]*(term.flux[fID] * Ef_mag)/delta
    # ap = term.sign[1]*(term.flux[fID] * Ef_mag)/delta

    # ap = term.sign*(term.flux[fID] * area)/delta1 # worked with explicit
    ap = term.sign*(term.flux[fID] * area)/delta # playing around
    
    # Increment sparse array
    # ac = -ap
    # an = ap
    
    # ap, ap*face_value # when using interpolated face value
    # ap, ap*values[pcellID] # when using neighbour cell value

    nzcellID = spindex(colptr, rowval, cellID, pcellID)

    # pzcellID = spindex(colptr, rowval, pcellID, pcellID)

    # pnzcellID = spindex(colptr, rowval, pcellID, cellID)

    # pnzcellID = spindex(colptr, rowval, cellID, pcellID)
    # nzcellID = spindex(colptr, rowval, pcellID, cellID)

    # if !bc.value.ismaster
    #     # Atomix.@atomic nzval[nzcellID] += ac
    #     # return an, 0.0
    #     return ac, an*values[pcellID]
    # end

    # Explicit version working!
    # Atomix.@atomic nzval[pzcellID] += -ap
    # Atomix.@atomic b[pcellID] += -ap*values[cellID] # explicit this works
    # Atomix.@atomic b[cellID] += -ap*values[pcellID] # explicit this works
    # # -ap, -ap*values[pcellID] # explicit this works
    # -ap, 0.0 # explicit this works

    # Explicit allowing looping over slave patch
    # -ap, -ap*values[pcellID] # explicit this works

    # Playing with implicit version
    # Atomix.@atomic nzval[pzcellID] += -ap
    # Atomix.@atomic nzval[pnzcellID] += ap
    Atomix.@atomic nzval[nzcellID] += ap
    -ap, 0.0
end

@define_boundary Periodic Divergence{Linear} begin

    # if !bc.value.ismaster
    #     return 0.0, 0.0
    # end

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
    
    weight = delta1/delta
    one_minus_weight = one(eltype(weight)) - weight
    # when using interpolated face value
    face_value = weight*values[cellID] + one_minus_weight*values[pcellID]

    # Calculate ap value to increment
    ap = 0.0
    ap = term.sign[1]*(term.flux[fID])
    # if bc.value.ismaster
    #     ap = -term.sign[1]*(term.flux[fID])
    # else
    #     ap = -term.sign[1]*(term.flux[fID]) # correct face normal
    # end

    ac = weight*ap
    an = one_minus_weight*ap

    nzcellID = spindex(colptr, rowval, cellID, pcellID)
    
    # pzcellID = spindex(colptr, rowval, pcellID, pcellID)
    # pnzcellID = spindex(colptr, rowval, pcellID, cellID)

    # pnzcellID = spindex(colptr, rowval, cellID, pcellID)
    # nzcellID = spindex(colptr, rowval, pcellID, cellID)

    # if !bc.value.ismaster
    #     # Atomix.@atomic nzval[nzcellID] += an
    #     return ac, -an*face_value
    #     # return 0.0, 0.0
    # end

    # Explicit version working
    # Atomix.@atomic nzval[pzcellID] += -ac
    # Atomix.@atomic b[pcellID] += an*values[cellID] # explicit this works!
    # Atomix.@atomic b[cellID] += -an*values[pcellID] # explicit this works!
    # # ac, -an*values[pcellID] # explicits this works!
    # ac, 0.0 # explicits this works!

    # Explicit allowing looping over slave patch
    # ac, -an*values[pcellID] # explicit this works

    # Playing around with implicit version
    # Atomix.@atomic nzval[pzcellID] += -ac
    Atomix.@atomic nzval[nzcellID] += an
    ac, 0.0
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