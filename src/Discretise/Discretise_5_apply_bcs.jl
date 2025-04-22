export apply_boundary_conditions!
export get_boundaries


apply_boundary_conditions!(eqnModel, BCs, component, time, config) = begin
    _apply_boundary_conditions!(eqnModel.model, BCs, eqnModel, component, time, config)
end

# Apply Boundaries Function
function _apply_boundary_conditions!(
    model::Model{TN,SN,T,S}, BCs::B, eqnModel, component, time, config) where {TN,SN,T,S,B}
    nTerms = length(model.terms)

    # backend = _get_backend(mesh)
    (; hardware) = config
    (; backend, workgroup) = hardware

    # Retriecve variables for function
    mesh = model.terms[1].phi.mesh
    A = _A(eqnModel)
    b = _b(eqnModel, component)

    # Deconstruct mesh to required fields
    (; faces, cells, boundary_cellsID) = mesh

    # Call sparse array field accessors
    colval = _colval(A)
    rowptr = _rowptr(A)
    nzval = _nzval(A)

    # Get user-defined integer types
    integer = _get_int(mesh)
    ione = one(integer)

    # Loop over boundary conditions to apply boundary conditions 
    boundaries_cpu = get_boundaries(mesh.boundaries)

    for BC ∈ BCs
        # Copy to CPU
        facesID_range = boundaries_cpu[BC.ID].IDs_range
        start_ID = facesID_range[1]

        # update user defined boundary storage (if needed)
        # update_user_boundary!(BC, faces, cells, facesID_range, time, config)
        #= The `model` passed here is defined in ModelFramework_0_types.jl line 87. It has two properties: terms and sources which define the equation being solved =#
        update_user_boundary!(
            BC, eqnModel, component, faces, cells, facesID_range, time, config)
        
        # Execute apply boundary conditions kernel
        kernel_range = length(facesID_range)

        kernel! = apply_boundary_conditions_kernel!(backend, workgroup, kernel_range)
        kernel!(
            model, BC, model.terms, faces, cells, start_ID, boundary_cellsID, colval, rowptr, nzval, b, ione, component, time, ndrange=kernel_range
            )
        # # KernelAbstractions.synchronize(backend)
    end
    # # KernelAbstractions.synchronize(backend)
    nothing
end

update_user_boundary!(
    BC::AbstractBoundary, eqnModel, component, faces, cells, facesID_range, time, config) = nothing

# Function to prevent redundant CPU copy

function get_boundaries(boundaries::Array)
    return boundaries
end

# Function to copy from GPU to CPU
function get_boundaries(boundaries::AbstractGPUArray)
    # Copy boundaries to CPU
    boundaries_cpu = Array{eltype(boundaries)}(undef, length(boundaries))
    copyto!(boundaries_cpu, boundaries)
    return boundaries_cpu
end

# Apply boundary conditions kernel definition
@kernel function apply_boundary_conditions_kernel!(
    model::Model{TN,SN,T,S}, BC, terms, 
    faces, cells, start_ID, boundary_cellsID, colval, rowptr, nzval, b, ione, component, time
    ) where {TN,SN,T,S}
    i = @index(Global)

    # Redefine thread index to correct starting ID 
    j = i + start_ID - 1
    fID = j

    # Retrieve workitem cellID, cell and face
    cellID = boundary_cellsID[j]
    face = faces[fID]
    cell = cells[cellID] 

    # zcellID = nzval_index(rowptr, colval, cellID, cellID, ione)
    zcellID = spindex(rowptr, colval, cellID, cellID)

    # Call apply generated function
    AP, BP = apply!(
        model, BC, terms, 
        colval, rowptr, nzval, cellID, zcellID, cell, face, fID, i, component, time
        )
    Atomix.@atomic nzval[zcellID] += AP
    Atomix.@atomic b[cellID] += BP
end

# Apply generated function definition
@generated function apply!(
    model::Model{TN,SN,T,S}, BC, terms, colval, rowptr, nzval, cellID, zcellID, cell, face, fID, i, component, time
    ) where {TN,SN,T,S}

    # Definition of main assignment loop (one per patch)
    func_calls = Expr[]
    for t ∈ 1:TN 
        call = quote
            ap, bp = BC(
                terms[$t], 
                colval, rowptr, nzval, cellID, zcellID, cell, face, fID, i, component, time
                )
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

