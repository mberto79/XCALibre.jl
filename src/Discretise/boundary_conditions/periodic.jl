export Periodic, PeriodicParent, PeriodicConnectivity
export construct_periodic
export LinearTransform
export adjust_boundary!
import XCALibre.ModelFramework._extend_matrix

abstract type AbstractPeriodic <: AbstractPhysicalConstraint end


"""
    struct Periodic{I,V,R<:UnitRange} <: AbstractPhysicalConstraint
        ID::I
        value::V
    end

Implicit implementation of `Periodic` boundary condition. Note that to apply this condition two periodic patch pairs need to be constructed using the function `construct_periodic`. The implementation currently requires conformant patch pairs.

### Fields
- `ID` is the name of the boundary given as a symbol (e.g. :inlet). Internally it gets replaced with the boundary index ID
- `value` tuple containing information needed to apply this boundary
"""
struct Periodic{I,V,R<:UnitRange} <: AbstractPeriodic
    ID::I 
    value::V
    IDs_range::R
end
Adapt.@adapt_structure Periodic

struct PeriodicParent{I,V,R<:UnitRange} <: AbstractPeriodic
    ID::I 
    value::V
    IDs_range::R
end
Adapt.@adapt_structure PeriodicParent

@kwdef struct PeriodicValue{I<:Integer,T,VI,B}
    patchID::I
    transform::T
    face_map::VI
    isparent::B
end
Adapt.@adapt_structure PeriodicValue

@kwdef struct LinearTransform{D<:Number, N<:AbstractVector}
    distance::D
    direction::N
end
Adapt.Adapt.@adapt_structure LinearTransform

adapt_value(value::PeriodicValue, mesh) = begin
    I = _get_int(mesh)
    F = _get_float(mesh)
    (; patchID, transform, face_map, isparent)  = value
    (; distance, direction) = transform
    @allowscalar PeriodicValue(
        I(patchID), 
        LinearTransform(F(distance), SVector{3,F}(direction)), 
        I.(face_map), 
        isparent
    )
end

struct PeriodicConnectivity{I}
    i::I
    j::I
end
Adapt.@adapt_structure PeriodicConnectivity

"""
    construct_periodic(mesh, backend, patch1::Symbol, patch2::Symbol)

Function for construction of periodic boundary conditions.

### Input
- `mesh` -- Mesh.
- `backend`  -- Backend configuraton.
- `patch1`  -- Primary periodic patch ID.
- `patch2`   -- Neighbour periodic patch ID.

### Output
- periodic::Tuple - tuple containing boundary definitions for `patch1` and `patch2` i.e. (periodic1, periodic2). The fields of `periodic1` and `periodic2` are 

    - `ID` -- Index to access boundary information in mesh object
    - `value` represents a `PeriodicValue` struct with the following fields:
        - index -- ID used to find boundary geometry information in the mesh object
        - distance -- perpendicular distance between the patches
        - face_map -- vector providing indeces to faces of match patch
        - ismaster -- flat to identify one of the patch pairs as the main patch

### Example
    - `periodic = construct_periodic(mesh, CPU(), :top, :bottom)` - Example using CPU 
    backend with periodic boundaries named `top` and `bottom`.

"""
function construct_periodic(
    transform::LinearTransform, mesh, backend, patch1::Symbol, patch2::Symbol)

    (; faces, boundaries) = mesh

    boundary_information = boundary_map(mesh)
    idx1 = boundary_index(boundary_information, patch1)
    idx2 = boundary_index(boundary_information, patch2)

    face1 = faces[boundaries[idx1].IDs_range[1]]
    face2 = faces[boundaries[idx2].IDs_range[1]]
    distance = abs((face1.centre - face2.centre)⋅face1.normal)
    isapprox(distance, transform.distance, atol=1e-10) || error(
        "distance given does not match patch distance within 1e-10 units (expected $distance)")

    distance = transform.distance # if user provided distance ok, use it

    # extract and check number of faces in each patch
    nfaces1 = length(boundaries[idx1].IDs_range)
    nfaces2 = length(boundaries[idx2].IDs_range)
    nfaces1 == nfaces2 || error("The number of faces for periodic patches should be equal")

    faceAddress1 = zeros(Int64, nfaces1)
    faceAddress2 = zeros(Int64, nfaces1)
    testData = zeros(Float64, nfaces1)

    faceCentres1 = getproperty.(faces[boundaries[idx1].IDs_range], :centre)
    patchTranslated1 = faceCentres1 .- [distance*face1.normal]
    ID_record = [1:nfaces1;] # use to keep record of face IDs (needed due to deleteat!)
    for (i, face) ∈ enumerate(faces[boundaries[idx2].IDs_range]) # use @view?
        testData = norm.(patchTranslated1 .- [face.centre])
        val, id = findmin(testData)
        idx = ID_record[id] # get the face index from the record (cannot use id directly!)
        fID = boundaries[idx1].IDs_range[i] # master face ID
        pfID = boundaries[idx2].IDs_range[idx] # periodic shadow face ID
        faceAddress1[i] = pfID
        faceAddress2[idx] = fID

        # # ensure face normals for master and shadow face are exact but flipped
        # println(face.normal)
        # # @reset face.normal = faces[fID].normal 
        # @reset face.normal = face.normal 
        # faces[pfID] = face

        # println("modified ", face.normal)

        # shrink search space to remove face pair already found 
        ID_record = deleteat!(ID_record, id)
        patchTranslated1 = deleteat!(patchTranslated1, id)
    end

    values1 = PeriodicValue(
        patchID=idx2, transform=transform, face_map=faceAddress1, isparent=true)
    values2 = PeriodicValue(
        patchID=idx1, transform=transform, face_map=faceAddress2, isparent=false)

    p1 = PeriodicParent(patch1, values1)
    p2 = Periodic(patch2, values2)

    periodic1 = adapt(backend, p1)
    periodic2 = adapt(backend, p2)
    
    return (periodic1, periodic2)
end

_extend_matrix(BC::PeriodicParent, mesh,  i, j) = begin
    i_ext, j_ext = periodic_matrix_connectivity(BC, mesh)
    return [i; i_ext], [j; j_ext]
end

function periodic_matrix_connectivity(BC::PeriodicParent, mesh)
    (; faces, boundaries) = mesh

    # Copy to CPU: this is a temporary fix and should be re-thought avoiding data transfer
    BC_cpu = adapt(CPU(), BC)
    faces_cpu = adapt(CPU(), faces)
    boundaries_cpu = adapt(CPU(), boundaries)
    BC1 = boundaries_cpu[BC.ID].IDs_range

    fmap1 = BC_cpu.value.face_map
    i = zeros(Int, 2*length(fmap1))
    j = zeros(Int, 2*length(fmap1))

    nindex = 0
    for (fID1, fID2) ∈ zip(BC1, fmap1)
        face1 = faces_cpu[fID1]
        face2 = faces_cpu[fID2]
        cID1 = face1.ownerCells[1]
        cID2 = face2.ownerCells[1]

        nindex += 1
        i[nindex] = cID1
        j[nindex] = cID2

        nindex += 1
        i[nindex] = cID2
        j[nindex] = cID1
    end

    return i, j
end

@define_boundary Periodic Laplacian{Linear} begin
    return 0.0, 0.0
end

# @define_boundary Union{PeriodicParent,Periodic} Laplacian{Linear} begin
@define_boundary PeriodicParent Laplacian{Linear} begin

    phi = term.phi
    mesh = phi.mesh 
    (; faces, cells) = mesh
    values = get_values(phi, component)
    (; transform) = bc.value
    
    (; area, normal, e) = face

    # determine id of periodic cell and interpolate face value
    pfID = bc.value.face_map[i] # id of periodic face 
    pface = faces[pfID]
    pcellID = pface.ownerCells[1]
    C1 = cell.centre
    C2 = cells[pcellID].centre - transform.direction*transform.distance

    # for improved accuracy this needs to include the discretisation used for noncorrection
    d = C2 - C1
    Δ = norm(d)
    Sf = area*normal # no need to flip normal direction - outwards by mesh contract
    Af = norm(Sf)
    
    
    # Use form below to ensure correctness, could be simplified for performance
    e = e # original
    Ef = ((Sf⋅Sf)/(Sf⋅e))*e # original
    Ef_mag = norm(Ef)
    gamma = -term.sign*(term.flux[fID]*Ef_mag)/Δ

    # ap = term.sign*(term.flux[fID]*Af)/Δ

    # Test formulation using vector d instead of e to explore any stability benefits
    # Ef = ((Sf⋅Sf)/(Sf⋅d))*d
    # Ef_mag = norm(Ef)
    # ap = term.sign*(term.flux[fID]*Ef_mag)/Δ
    
    # Increment sparse array
    # ac = -ap
    # an = ap

    # NN = spindex(rowptr, colval, pcellID, pcellID)
    # nzval[NN] += ac

    # NP = spindex(rowptr, colval, pcellID, cellID)
    # nzval[NP] += an

    NN = spindex(rowptr, colval, pcellID, pcellID)
    Atomix.@atomic nzval[NN] += gamma

    NP = spindex(rowptr, colval, pcellID, cellID)
    Atomix.@atomic nzval[NP] += -gamma

    PN = spindex(rowptr, colval, cellID, pcellID)
    Atomix.@atomic nzval[PN] += -gamma

    return gamma, 0.0 # PP assigned first value returned
end

@define_boundary Periodic Divergence{Linear} begin
    0.0, 0.0
end

@define_boundary PeriodicParent Divergence{Linear} begin
    phi = term.phi
    mesh = phi.mesh 
    (; faces, cells) = mesh
    values = get_values(phi, component)
    (; transform) = bc.value

    # determine id of periodic cell and interpolate face value
    pfID = bc.value.face_map[i] # id of periodic face 
    pface = faces[pfID]
    pcellID = pface.ownerCells[1]

    w = pface.delta/(face.delta + pface.delta)

    # Calculate link coefficients
    term.flux[pfID] = -term.flux[fID] # copy flux from master to shadow (for stability)
    ap = term.sign*(term.flux[fID])
    ac = ap*w
    an = ap*(one(w) - w)

    NN = spindex(rowptr, colval, pcellID, pcellID)
    NP = spindex(rowptr, colval, pcellID, cellID)
    PN = spindex(rowptr, colval, cellID, pcellID)
    
    # handle shadow cell first
    Atomix.@atomic nzval[NN] += -an 
    Atomix.@atomic nzval[NP] += -ac
    # pos neg works

    # now handle master cell 
    Atomix.@atomic nzval[PN] += an
    return ac, 0.0
end

@define_boundary Periodic Divergence{Upwind} begin
    0.0, 0.0
end
# @define_boundary Union{PeriodicParent,Periodic} Divergence{Upwind} begin
@define_boundary PeriodicParent Divergence{Upwind} begin
    phi = term.phi
    mesh = phi.mesh 
    (; faces, cells) = mesh
    values = get_values(phi, component)
    (; transform) = bc.value

    # determine id of periodic cell and interpolate face value
    pfID = bc.value.face_map[i] # id of periodic face 
    pface = faces[pfID]
    pcellID = pface.ownerCells[1]

    term.flux[pfID] = -term.flux[fID] # copy flux from master to shadow (for stability)
    mdot = term.sign*(term.flux[fID])
    ap = max(mdot, 0.0) # flow leaves master
    an = -max(-mdot, 0.0) # flow leaves shadow

    NN = spindex(rowptr, colval, pcellID, pcellID)
    NP = spindex(rowptr, colval, pcellID, cellID)
    PN = spindex(rowptr, colval, cellID, pcellID)
    
    # handle shadow cell first
    Atomix.@atomic nzval[NN] += -an
    Atomix.@atomic nzval[NP] += -ap

    # now handle master cell 
    Atomix.@atomic nzval[PN] += an
    return ap, 0.0
end

@define_boundary Periodic Divergence{LUST} begin
    0.0, 0.0
end

@define_boundary PeriodicParent Divergence{LUST} begin
    phi = term.phi
    mesh = phi.mesh 
    (; faces, cells) = mesh
    values = get_values(phi, component)
    (; transform) = bc.value

    # determine id of periodic cell and interpolate face value
    pfID = bc.value.face_map[i] # id of periodic face 
    pface = faces[pfID]
    pcellID = pface.ownerCells[1]


    # Calculate interpoloation weight
    w = pface.delta/(face.delta + pface.delta)

    # Calculate link coefficients
    mdot = term.sign*(term.flux[fID])
    term.flux[pfID] = -term.flux[fID] # copy flux from master to shadow (for stability)
    acLinear = mdot*w 
    anLinear = mdot*(one(w) - w)
    acUpwind = max(mdot, 0.0) 
    anUpwind = -max(-mdot, 0.0)
    ac = 0.75*acLinear + 0.25*acUpwind
    an = 0.75*anLinear + 0.25*anUpwind

    NN = spindex(rowptr, colval, pcellID, pcellID)
    NP = spindex(rowptr, colval, pcellID, cellID)
    PN = spindex(rowptr, colval, cellID, pcellID)
    
    # handle shadow cell first
    Atomix.@atomic nzval[NN] += -an
    Atomix.@atomic nzval[NP] += -ac

    # now handle master cell 
    Atomix.@atomic nzval[PN] += an
    return ac, 0.0
end

@define_boundary Union{PeriodicParent,Periodic} Si begin
    0.0, 0.0
end