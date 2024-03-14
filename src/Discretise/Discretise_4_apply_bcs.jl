export apply_boundary_conditions!

apply_boundary_conditions!(eqn, BCs) = begin
    _apply_boundary_conditions!(eqn.model, BCs, eqn)
end

function _apply_boundary_conditions!(
    model::Model{TN,SN,T,S}, BCs::B, eqn) where {T,S,TN,SN,B}
    nTerms = length(model.terms)

    # Define variables for function
    mesh = model.terms[1].phi.mesh
    A = _A(eqn)
    b = _b(eqn)
    (; boundaries, faces, cells, boundary_cellsID) = mesh

    # Deconstruct sparse array dependent on sparse arrays type
    rowval = _rowval(A)
    colptr = _colptr(A)
    nzval = _nzval(A)

    # Get types and create integer(one)
    backend = _get_backend(mesh)
    integer = _get_int(mesh)
    ione = one(integer)

    # Execute function to apply boundary conditions for all terms and boundaries
    # for bci in 1:length(BCs)
    #     for t in 1:nTerms
    #         Execute_apply_boundary_condition_kernel!(BCs[bci], model.terms[t], 
    #                                                 backend, boundaries, faces, cells,
    #                                                 boundary_cellsID, ione, rowval_array,
    #                                                 colptr_array, nzval_array, b_array)
    #         KernelAbstractions.synchronize(backend)
    #     end
    # end
    for BC ∈ BCs
        CUDA.@allowscalar start_ID = mesh.boundaries[BC.ID].IDs_range[1]
        CUDA.@allowscalar facesID_range = mesh.boundaries[BC.ID].IDs_range
        kernel! = apply_boundary_conditions_kernel!(backend)
        kernel!(
            model, BC, model.terms, faces, cells, start_ID, boundary_cellsID, rowval, colptr, nzval, b, ione, ndrange=length(facesID_range)
            )
        KernelAbstractions.synchronize(backend)
    end
    nothing
    # end
end

   # Unpack terms that make up the model (not sources)
    # nTerms = model.parameters[3]
    # nTerms = TN

    # # Definition of main assignment loop (one per patch)
    # assignment_loops = []
    # for bci ∈ 1:length(BCs.parameters)
    #     func_calls = Expr[]
    #     for t ∈ 1:nTerms 
    #         call = quote
    #             # (BCs[$bci])(model.terms[$t], A, b, cellID, cell, face, faceID)
    #             Execute_apply_boundary_condition_kernel!(BCs[$bci], model.terms[$t], backend, boundaries, faces, cells, boundary_cellsID, A, b)
    #         end
    #         push!(func_calls, call)
    #     end
    #     assignment_loop = quote
    #         # (; IDs_range) = boundaries[BCs[$bci].ID]
    #         # @inbounds for i ∈ IDs_range
    #         #     faceID = i
    #         #     cellID = boundary_cellsID[i]
    #         #     face = faces[faceID]
    #         #     cell = cells[cellID]
    #             $(func_calls...)
    #         # end
    #     end
    #     push!(assignment_loops, assignment_loop.args...)
    # end

    # quote
    # Extract number of terms

@kernel function apply_boundary_conditions_kernel!(
    model::Model{TN,SN,T,S}, BC, terms, 
    faces, cells, start_ID, boundary_cellsID, rowval, colptr, nzval, b, ione
    ) where {TN,SN,T,S}

    i = @index(Global)
    # (; IDs_range) = boundaries[BC.ID]
    j = i + start_ID - 1
    # i = BC.ID
    # (; IDs_range) = boundaries[i]
    
    faceID = j
    cellID = boundary_cellsID[j]
    face = faces[faceID]
    cell = cells[cellID] 
    apply!(model, BC, terms, rowval, colptr, nzval, b, cellID, cell, face, faceID, ione)
end

@generated function apply!(
    model::Model{TN,SN,T,S}, BC, terms, 
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione) where {TN,SN,T,S}

    # Definition of main assignment loop (one per patch)
    func_calls = Expr[]
    for t ∈ 1:TN 
        call = quote
            (BC)(terms[$t], rowval, colptr, nzval, b, cellID, cell, face, fID, ione)
        end
        push!(func_calls, call)
    end
    quote
        $(func_calls...)
    end
end

@generated function boundary_indices(mesh::M, BCs::B) where {M<:AbstractMesh,B}
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