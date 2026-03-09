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

@kwdef struct LinearTransform{T<:AbstractVector}
    distance::T
end
Adapt.Adapt.@adapt_structure LinearTransform

adapt_value(value::PeriodicValue, mesh) = begin
    I = _get_int(mesh)
    F = _get_float(mesh)
    (; patchID, transform, face_map, isparent)  = value
    (; distance) = transform
    @allowscalar PeriodicValue(
        I(patchID), 
        LinearTransform(SVector{3,F}(distance)), 
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
    construct_periodic(mesh, backend, patch1::Symbol, patch2::Symbol; tol=1e-8)

Function for construction of periodic boundary conditions.

### Input
- `mesh` -- Mesh.
- `backend`  -- Backend configuraton.
- `patch1`  -- Primary periodic patch ID.
- `patch2`   -- Neighbour periodic patch ID.
- `tol` -- keyword argument providing the tolerance used to find matching face pairs

### Output
- periodic::Tuple - tuple containing boundary definitions for `patch1` and `patch2` i.e. (periodic1, periodic2). The fields of `periodic1` and `periodic2` are 

    - `ID` -- Index to access boundary information in mesh object
    - `value` represents a `PeriodicValue` struct with the following fields:
        - patchID -- boundary/patch ID
        - transform -- stores information to apply the patch pair matching e.g. LinearTransform.
        - face_map -- vector providing indices to faces of match patch
        - ismaster -- flat to identify one of the patch pairs as the main patch

### Example
    - `periodic = construct_periodic(mesh, CPU(), :top, :bottom)` - Example using CPU 
    backend with periodic boundaries named `top` and `bottom`.

"""
function construct_periodic(mesh, backend, patch1::Symbol, patch2::Symbol; tol=1e-7)

    (; faces, boundaries) = mesh

    # Get Patch Indices
    boundary_info = boundary_map(mesh)
    idx1 = boundary_index(boundary_info, patch1)
    idx2 = boundary_index(boundary_info, patch2)
    
    global_ids1 = boundaries[idx1].IDs_range
    global_ids2 = boundaries[idx2].IDs_range

    nfaces = length(global_ids1)
    length(global_ids2) == nfaces || error(
        "Periodic mismatch: Patch $patch1 has $(length(global_ids1)) faces, but $patch2 has $(length(global_ids2))")

    # Extract Centers
    centers1 = [faces[id].centre for id in global_ids1]
    centers2 = [faces[id].centre for id in global_ids2]

    # Calculate Translation Vector (Difference between centroids)
    center_mass1 = mean(centers1)
    center_mass2 = mean(centers2)
    translation_vec = center_mass2 - center_mass1 
    @info "Translating patches by $translation_vec"
    
    # Shift Patch 1 to overlap Patch 2 to define the "Target" points for Patch 2
    targets = centers1 .+ Ref(translation_vec)

    # Find largest search direction
    min_bound = minimum(targets)
    max_bound = maximum(targets)
    extent = max_bound - min_bound
    
    # Find the index with the largest extent
    sort_axis = argmax([extent[1], extent[2], extent[3]])
    
    # Sort both lists by the largest axis by extent
    # p_idx stores the permutation to get back to original order
    p1_idx = sortperm(targets,  by = p -> p[sort_axis])
    p2_idx = sortperm(centers2, by = p -> p[sort_axis])
    
    # Create sorted views for the search
    targets_sorted = targets[p1_idx]
    sources_sorted = centers2[p2_idx]

    faceAddress1 = zeros(Int64, nfaces)
    faceAddress2 = zeros(Int64, nfaces)
    
    # Sliding Window Search
    search_start = 1
    for i in 1:nfaces
        target_pt = targets_sorted[i]
        target_val = target_pt[sort_axis]
        
        # Advance the window start
        while search_start <= nfaces && sources_sorted[search_start][sort_axis] < target_val - tol
            search_start += 1
        end
        
        # Contral and store variables
        best_dist = Inf
        best_match_idx = -1
        
        # Check all candidates in the sorted window
        for j in search_start:nfaces
            source_pt = sources_sorted[j]
            axis_dist = source_pt[sort_axis] - target_val
            
            # If the axis distance alone exceeds tolerance, 
            # no further points in the sorted list can possibly be matches.
            if axis_dist > tol
                break
            end
            
            # Calculate 3D distance
            dist = norm(target_pt - source_pt)
            
            # Logic: We want the CLOSEST match within tolerance
            if dist < best_dist
                best_dist = dist
                best_match_idx = j
            end
        end
        
        # Verification Step
        if best_match_idx != -1 && best_dist < tol
             # MATCH FOUND: Map the IDs
             orig_local_idx1 = p1_idx[i]
             orig_local_idx2 = p2_idx[best_match_idx]
             
             global_id1 = global_ids1[orig_local_idx1]
             global_id2 = global_ids2[orig_local_idx2]
             
             faceAddress1[orig_local_idx1] = global_id2
             faceAddress2[orig_local_idx2] = global_id1
        else
            error("Periodic mismatch at Face $i. \nTarget: $target_pt \nNo match found closer than tolerance $tol")
        end
        
        # Map back to Global IDs      
        orig_local_idx1 = p1_idx[i]
        orig_local_idx2 = p2_idx[best_match_idx]
        
        global_id1 = global_ids1[orig_local_idx1]
        global_id2 = global_ids2[orig_local_idx2]
        
        faceAddress1[orig_local_idx1] = global_id2
        faceAddress2[orig_local_idx2] = global_id1
    end

    # Construct periodic boundaries and adapt to target backend
    transform = LinearTransform(translation_vec) # transformation applied
    values1 = PeriodicValue(
        patchID=idx2, transform=transform, face_map=faceAddress1, isparent=true)
    values2 = PeriodicValue(
        patchID=idx1, transform=transform, face_map=faceAddress2, isparent=false)

    periodic1 = adapt(backend, PeriodicParent(patch1, values1))
    periodic2 = adapt(backend, Periodic(patch2, values2))
    
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
    C2 = cells[pcellID].centre - transform.distance

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
    C1 = cell.centre
    C2 = cells[pcellID].centre - transform.distance
    Cf = face.centre
    n = face.normal

    Pf = Cf - C1
    PN = C2 - C1 

    wn = (Pf⋅n)/(PN⋅n)
    w = one(wn) - wn
    # w = pface.delta/(face.delta + pface.delta)
    # wn = one(w) - w

    # Calculate link coefficients
    ap = term.sign*(term.flux[fID])
    ac = ap*w
    an = ap*wn

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
    C1 = cell.centre
    C2 = cells[pcellID].centre - transform.distance
    Cf = face.centre
    n = face.normal

    Pf = Cf - C1
    PN = C2 - C1 

    wn = (Pf⋅n)/(PN⋅n)
    w = one(wn) - wn

    # Calculate link coefficients
    mdot = term.sign*(term.flux[fID])
    acLinear = mdot*w 
    anLinear = mdot*wn
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