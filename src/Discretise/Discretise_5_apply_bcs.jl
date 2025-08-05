export apply_boundary_conditions!



apply_boundary_conditions!(eqn, discretisation, BCs, component, time) = begin
    _apply_boundary_conditions!(eqn, discretisation, BCs, component, time)
end

# Apply Boundaries Function
function _apply_boundary_conditions!(
    eqn, discretisation, BCs::B, component, time) where B
    # nTerms = length(model.terms)

    (; hardware) = get_configuration(CONFIG)
    (; backend, workgroup) = hardware

    # Retriecve variables for function
    mesh = eqn.mesh
    A = _A(eqn)
    b = _b(eqn, component)

    # Deconstruct mesh to required fields
    (; faces, cells, boundary_cellsID) = mesh

    # Call sparse array field accessors
    colval = _colval(A)
    rowptr = _rowptr(A)
    nzval = _nzval(A)

    # Test implementation looking over all boundary faces 
    nbfaces = length(mesh.boundary_cellsID)

    for BC ∈ BCs
        facesID_range = BC.IDs_range
        # start_ID = facesID_range[1]

        # update user defined boundary storage (if needed)
        update_user_boundary!(BC, faces, cells, facesID_range, time)
        
    end
        # Execute apply boundary conditions kernel
        # ndrange = nbfaces
        # apply_bcs = apply_boundary_conditions_kernel!(
        #     _setup(backend, workgroup, ndrange)...)
        # apply_bcs(
        #     model, BCs,model.terms, faces, cells, boundary_cellsID, colval, rowptr, nzval, b, component, time, ndrange=ndrange
        #     )
        # KernelAbstractions.synchronize(backend)

        apply_BCs = discretisation.apply_BCs

        ndrange = nbfaces
        kernel! = apply_boundary_conditions_kernel!(_setup(backend, workgroup, ndrange)...)
        kernel!(
            apply_BCs, BCs, faces, cells, boundary_cellsID, colval, rowptr, nzval, b, component, time, ndrange=ndrange
            )
        KernelAbstractions.synchronize(backend)

    # Loop over boundary conditions to apply boundary conditions 
    # for BC ∈ BCs
    #     facesID_range = BC.IDs_range
    #     start_ID = facesID_range[1]

    #     # update user defined boundary storage (if needed)
    #     # update_user_boundary!(BC, faces, cells, facesID_range, time)
    #     #= The `model` passed here is defined in ModelFramework_0_types.jl line 87. It has two properties: terms and sources which define the equation being solved =#
    #     update_user_boundary!(BC, faces, cells, facesID_range, time)
        
    #     # Execute apply boundary conditions kernel
    #     kernel_range = length(facesID_range)

    #     kernel! = apply_boundary_conditions_kernel!(backend, workgroup, kernel_range)
    #     kernel!(
    #         model, BC, model.terms, faces, cells, start_ID, boundary_cellsID, colval, rowptr, nzval, b, component, time, ndrange=kernel_range
    #         )
    # end
end

update_user_boundary!(
    BC::AbstractBoundary, faces, cells, facesID_range, time) = nothing

# Apply boundary conditions kernel definition
# Experimental implementation 

@kernel function apply_boundary_conditions_kernel!(
    apply_BCs, BCs, faces, cells, boundary_cellsID, colval, rowptr, nzval, b, component, time
    )
    fID = @index(Global)

    calculate_coefficients(
        apply_BCs, BCs, faces, cells, boundary_cellsID,colval, rowptr, nzval, b, component, time, fID)
end

@generated function calculate_coefficients(
    apply_BCs, BCs, faces, cells, boundary_cellsID,colval, rowptr, nzval, b, component, time, fID)
    N = length(BCs.parameters)
    unroll = Expr(:block)
    for bci ∈ 1:N
        BC_checks = quote
            @inbounds begin
                BC = BCs[$bci] 
                (; start, stop) = BC.IDs_range
                if start <= fID <= stop
                    i = fID - start + 1
                    cellID = boundary_cellsID[fID]
                    face = faces[fID]
                    cell = cells[cellID] 

                    zcellID = spindex(rowptr, colval, cellID, cellID)
                    # AP, BP = apply!(
                    #     model, BC, terms, 
                    #     colval, rowptr, nzval, cellID, zcellID, cell, face, fID, i, component, time
                    #     )

                    AP, BP = apply_BCs(BC, colval, rowptr, nzval, cellID, zcellID, cell, face, fID, i, component, time)
                    Atomix.@atomic nzval[zcellID] += AP
                    Atomix.@atomic b[cellID] += BP
                    return nothing
                end
            end
        end
        push!(unroll.args, BC_checks)
    end
    return unroll
end

@generated function get_BC(BCs, index)
    N = length(BCs.parameters)
    exprs = Expr(:block)
    for i ∈ 1:N
        ex = quote
            if index == $i
                @inbounds BC = BCs[$i]
                (; start, stop) = BC.IDs_range
                return BC, start, stop
            end
        end
        push!(exprs.args, ex)
    end
    return exprs
end



# Current implementation 

# @kernel function apply_boundary_conditions_kernel!(
#     model::Model{TN,SN,T,S}, BC, terms, 
#     faces, cells, start_ID, boundary_cellsID, colval, rowptr, nzval, b, component, time
#     ) where {TN,SN,T,S}
#     i = @index(Global)

#     # Redefine thread index to correct starting ID 
#     j = i + start_ID - 1
#     fID = j

#     # Retrieve workitem cellID, cell and face
#     cellID = boundary_cellsID[j]
#     face = faces[fID]
#     cell = cells[cellID] 

#     zcellID = spindex(rowptr, colval, cellID, cellID)

#     # Call apply generated function
#     AP, BP = apply!(
#         model, BC, terms, 
#         colval, rowptr, nzval, cellID, zcellID, cell, face, fID, i, component, time
#         )
#     Atomix.@atomic nzval[zcellID] += AP
#     Atomix.@atomic b[cellID] += BP
# end

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

