export apply_boundary_conditions!

apply_boundary_conditions!(eqn, BCs) = begin
    _apply_boundary_conditions!(eqn.model, BCs, eqn)
end

@generated function _apply_boundary_conditions!(
    model::Model{TN,SN,T,S}, BCs::B, eqn) where {T,S,TN,SN,B}

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

    quote
    nTerms = TN
    (; A, b) = eqn.equation
    mesh = model.terms[1].phi.mesh
    (; boundaries, faces, cells, boundary_cellsID) = mesh
    precon = eqn.preconditioner

    rowval_array = rowval(A)
    colptr_array = colptr(A)
    nzval_array = nzval(A)

    backend = _get_backend(mesh)

    integer = _get_int(mesh)
    ione = one(integer)

    for bci in 1:length(BCs)
        for t in 1:nTerms
            Execute_apply_boundary_condition_kernel!(BCs[bci], model.terms[t], 
                                                    backend, boundaries, faces, cells,
                                                    boundary_cellsID, ione, rowval_array,
                                                    colptr_array, nzval_array, b)
            KernelAbstractions.synchronize(backend)
        end
    end

    # check_for_precon!(nzval_array, precon, backend)
    # $(assignment_loops...)
    nothing
    end
end

@generated function boundary_indices(mesh::M, BCs::B) where {M<:Mesh2,B}
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