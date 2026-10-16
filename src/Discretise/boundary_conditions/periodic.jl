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

    # empty patch pair (distributed rank owning no periodic faces) → no-op BCs
    if nfaces == 0
        F = _get_float(mesh)
        transform = LinearTransform(SVector{3,F}(0, 0, 0))
        values1 = PeriodicValue(
            patchID=idx2, transform=transform, face_map=_get_int(mesh)[], isparent=true)
        values2 = PeriodicValue(
            patchID=idx1, transform=transform, face_map=_get_int(mesh)[], isparent=false)
        return (adapt(backend, PeriodicParent(patch1, values1)),
                adapt(backend, Periodic(patch2, values2)))
    end

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

    faceAddress1 = zeros(_get_int(mesh), nfaces)
    faceAddress2 = zeros(_get_int(mesh), nfaces)
    
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
    i = zeros(_get_int(mesh), 2*length(fmap1))
    j = zeros(_get_int(mesh), 2*length(fmap1))

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

# the parent face and cell of the pair, and the cell across, seen from either patch
@inline function _periodic_pair(bc::AbstractPeriodic, faces, cellID, fID, i)
    pfID = bc.value.face_map[i] # face on the other side
    pcellID = faces.ownerCells[pfID][1]
    bc isa PeriodicParent ? (fID, cellID, pcellID) : (pfID, pcellID, cellID)
end

# Both patches take the parent face's coefficients; each writes only its own row (one writer per
# entry). Parent cell P, cell C across: row P gets A[P,P] += ac, A[P,C] += an; row C gets
# A[C,C] += -an, A[C,P] += -ac. Returns this side's diagonal (b gets nothing).
@inline function _periodic_row!(bc::AbstractPeriodic, nzval, rowptr, colval, cellID, fP, cP, cC, ac, an)
    if bc isa PeriodicParent
        nzval[spindex(rowptr, colval, cellID, cC)] += an
        return ac, zero(ac)
    else
        nzval[spindex(rowptr, colval, cellID, cP)] += -ac
        return -an, zero(an)
    end
end

# linear interpolation weights of the parent cell (w) and the cell across (wn) at the parent face
@inline function _periodic_weights(bc, cells, faces, fP, cP, cC)
    C1 = cells.centre[cP]
    C2 = cells.centre[cC] - bc.value.transform.distance
    n = faces.normal[fP]
    wn = ((faces.centre[fP] - C1)⋅n)/((C2 - C1)⋅n)
    one(wn) - wn, wn
end

@define_boundary Union{PeriodicParent,Periodic} Laplacian{Linear} begin
    fP, cP, cC = _periodic_pair(bc, faces, cellID, fID, i)
    (; transform) = bc.value
    area, normal, e = faces.area[fP], faces.normal[fP], faces.e[fP]

    # for improved accuracy this needs to include the discretisation used for noncorrection
    d = (cells.centre[cC] - transform.distance) - cells.centre[cP]
    Δ = norm(d)
    Sf = area*normal # no need to flip normal direction - outwards by mesh contract

    # Use form below to ensure correctness, could be simplified for performance
    Ef = ((Sf⋅Sf)/(Sf⋅e))*e
    Ef_mag = norm(Ef)
    gamma = -term.sign*(term.flux[fP]*Ef_mag)/Δ

    _periodic_row!(bc, nzval, rowptr, colval, cellID, fP, cP, cC, gamma, -gamma)
end

@define_boundary Union{PeriodicParent,Periodic} Divergence{Linear} begin
    fP, cP, cC = _periodic_pair(bc, faces, cellID, fID, i)
    w, wn = _periodic_weights(bc, cells, faces, fP, cP, cC)

    # Calculate link coefficients
    ap = term.sign*(term.flux[fP])
    _periodic_row!(bc, nzval, rowptr, colval, cellID, fP, cP, cC, ap*w, ap*wn)
end

@define_boundary Union{PeriodicParent,Periodic} Divergence{Upwind} begin
    fP, cP, cC = _periodic_pair(bc, faces, cellID, fID, i)

    mdot = term.sign*(term.flux[fP])
    z = zero(mdot)
    ap = max(mdot, z) # flow leaves master
    an = -max(-mdot, z) # flow leaves shadow
    _periodic_row!(bc, nzval, rowptr, colval, cellID, fP, cP, cC, ap, an)
end

@define_boundary Union{PeriodicParent,Periodic} Divergence{LUST} begin
    fP, cP, cC = _periodic_pair(bc, faces, cellID, fID, i)
    w, wn = _periodic_weights(bc, cells, faces, fP, cP, cC)

    # Calculate link coefficients
    mdot = term.sign*(term.flux[fP])
    acLinear = mdot*w 
    anLinear = mdot*wn
    z = zero(mdot)
    acUpwind = max(mdot, z)
    anUpwind = -max(-mdot, z)
    three_quarters = TF(0.75)
    quarter = TF(0.25)
    ac = three_quarters*acLinear + quarter*acUpwind
    an = three_quarters*anLinear + quarter*anUpwind
    _periodic_row!(bc, nzval, rowptr, colval, cellID, fP, cP, cC, ac, an)
end

# LUSTLU: LUST coefficients plus the explicit correction of the upwind quarter. Both sides take the
# flux of the parent face, and each adds the correction to its own cell (the kernel writes b only there)
@inline function (bc::AbstractPeriodic)(
    term::Operator{F,P,I,Divergence{LUSTLU}}, colval, rowptr, nzval, cellID, zcellID,
    cells::AbstractVector{<:Cell{TF}}, faces, fID, i, component, time
    ) where {F,P,I,TF}
    ap, bp = bc(_lust(term), colval, rowptr, nzval, cellID, zcellID, cells, faces, fID, i, component, time)
    ap, bp + _periodic_lustlu_source(bc, term, term.grad, cells, faces, cellID, fID, i, component)
end

_periodic_lustlu_source(bc, term, grad::Nothing, cells::AbstractVector{<:Cell{TF}}, faces, cellID, fID, i, component) where TF = zero(TF)

@inline function _periodic_lustlu_source(
    bc, term, grad, cells::AbstractVector{<:Cell{TF}}, faces, cellID, fID, i, component) where TF
    pfID = bc.value.face_map[i] # face on the other side
    pcellID = faces.ownerCells[pfID][1]
    isparent = bc isa PeriodicParent
    fP, cP, fC, cC = isparent ? (fID, cellID, pfID, pcellID) : (pfID, pcellID, fID, cellID)
    flux = term.sign*term.flux[fP] # out of the parent cell
    fU, uID = flux > zero(flux) ? (fP, cP) : (fC, cC)
    δ = _component(_reconstruct(grad[uID], faces.centre[fU] - cells.centre[uID]), component)
    source = -flux*TF(0.25)*δ # into the parent cell
    isparent ? source : -source
end

@inline _component(δ, ::Nothing) = δ
@inline _component(δ, component) = δ[component.value]

@define_boundary Union{PeriodicParent,Periodic} Si begin
    0.0, 0.0
end
