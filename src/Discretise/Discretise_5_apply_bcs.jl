export apply_boundary_conditions!
export get_boundaries


apply_boundary_conditions!(eqn, BCs, component, config) = begin
    _apply_boundary_conditions!(eqn.model, BCs, eqn, component, config)
end

# Apply Boundaries Function
function _apply_boundary_conditions!(
    model::Model{TN,SN,T,S}, BCs::B, eqn, component,config) where {TN,SN,T,S,B}
    nTerms = length(model.terms)

    # backend = _get_backend(mesh)
    (; hardware) = config
    (; backend, workgroup) = hardware

    # Retriecve variables for function
    mesh = model.terms[1].phi.mesh
    A = _A(eqn)
    b = _b(eqn, component)

    # Deconstruct mesh to required fields
    (; faces, cells, boundary_cellsID) = mesh

    # Call sparse array field accessors
    rowval = _rowval(A)
    colptr = _colptr(A)
    nzval = _nzval(A)

    # Get user-defined integer types
    integer = _get_int(mesh)
    ione = one(integer)

    # Loop over boundary conditions to apply boundary conditions 
    for BC ∈ BCs
        # Copy to CPU
        facesID_range = get_boundaries(BC, mesh.boundaries)
        start_ID = facesID_range[1]

        # Execute apply boundary conditions kernel
        kernel_range = length(facesID_range)
        kernel! = apply_boundary_conditions_kernel!(backend, workgroup, kernel_range)
        kernel!(
            model, BC, model.terms, faces, cells, start_ID, boundary_cellsID, rowval, colptr, nzval, b, ione, component, ndrange=kernel_range
            )
        KernelAbstractions.synchronize(backend)
    end
    # KernelAbstractions.synchronize(backend)
    nothing
end

# Function to prevent redundant CPU copy
function get_boundaries(BC, boundaries::Array)
    facesID_range = boundaries[BC.ID].IDs_range
    return facesID_range
end

# Function to copy from GPU to CPU
# function get_boundaries(BC, boundaries::CuArray)
function get_boundaries(BC, boundaries::AbstractGPUArray)
    # Copy boundaries to CPU
    boundaries_cpu = Array{eltype(boundaries)}(undef, length(boundaries))
    copyto!(boundaries_cpu, boundaries)
    facesID_range = boundaries_cpu[BC.ID].IDs_range
    return facesID_range
end

# Apply boundary conditions kernel definition
@kernel function apply_boundary_conditions_kernel!(
    model::Model{TN,SN,T,S}, BC, terms, 
    faces, cells, start_ID, boundary_cellsID, rowval, colptr, nzval, b, ione, component
    ) where {TN,SN,T,S}
    i = @index(Global)

    # Redefine thread index to correct starting ID 
    j = i + start_ID - 1
    faceID = j

    # Retrieve workitem cellID, cell and face
    cellID = boundary_cellsID[j]
    face = faces[faceID]
    cell = cells[cellID] 

    zcellID = nzval_index(colptr, rowval, cellID, cellID, ione)

    # Call apply generated function
    AP, BP = apply!(model, BC, terms, cellID, zcellID, cell, face, faceID, i, component)
    Atomix.@atomic nzval[zcellID] += AP
    Atomix.@atomic b[cellID] += BP
end

# Apply generated function definition
@generated function apply!(
    model::Model{TN,SN,T,S}, BC, terms, cellID, zcellID, cell, face, fID, i, component
    ) where {TN,SN,T,S}

    # Definition of main assignment loop (one per patch)
    func_calls = Expr[]
    for t ∈ 1:TN 
        call = quote
            ap, bp = (BC)(terms[$t], cellID, zcellID, cell, face, fID, i, component)
            AP += ap
            BP += bp
        end
        push!(func_calls, call)
    end
    quote
        AP = 0.0
        BP = 0.0
        $(func_calls...)
        return AP, BP
    end
end

# Boundary indices generated function definition
@generated function boundary_indices(mesh::M, BCs::B) where {M<:AbstractMesh,B}

    # Definition of main boundary indices loop (one per patch)
    unpacked_BCs = []
    for i ∈ 1:length(BCs.parameters)
        unpack = quote
            name = BCs[$i].name
            index = boundary_index(boundaries, name)
            BC_indices = (BC_indices..., index)
        end
        push!(unpacked_BCs, unpack)
    end
    quote
        boundaries = mesh.boundaries
        BC_indices = ()
        $(unpacked_BCs...)
        return BC_indices
    end
end

