export discretise!, _nzval, _rowval, _colptr

discretise!(eqn, prev, runtime) = _discretise!(eqn.model, eqn, prev, runtime)

function _discretise!(
    model::Model{TN,SN,T,S}, eqn, prev, runtime
    ) where {TN,SN,T,S}

    # nTerms = TN
    # nSources = SN

    # assignment_block_1 = Expr[] # Ap
    # assignment_block_2 = Expr[] # An or b
    # assignment_block_3 = Expr[] # b (sources)

    # # Loop for operators
    # for t ∈ 1:nTerms
    #     function_call = quote
    #         scheme!(
    #             model.terms[$t], nzval, cell, face, cellN, ns, cIndex, nIndex, fID, prev, runtime
    #             )
    #     end
    #     push!(assignment_block_1, function_call)

    #     assign_source = quote
    #         scheme_source!(model.terms[$t], b, nzval, cell, cID, cIndex, prev, runtime)
    #     end
    #     push!(assignment_block_2, assign_source)
    # end

    # # Loop for sources
    # for s ∈ 1:nSources
    #     add_source = quote
    #         (; field, sign) = model.sources[$s]
    #         b[cID] += sign*field[cID]*volume
    #     end
    #     push!(assignment_block_3, add_source)
    # end

    # quote
    #     (; A, b) = eqn.equation
    #     mesh = model.terms[1].phi.mesh
    #     integer = _get_int(mesh)
    #     float = _get_float(mesh)
    #     backend = _get_backend(mesh)
    #     # (; faces, cells, ) = mesh
    #     (; faces, cells, cell_faces, cell_neighbours, cell_nsign) = mesh
    #     # (; rowval, colptr, nzval) = A
    #     rowval, colptr, nzval = sparse_array_deconstructor(A)
    #     fzero = zero(float) # replace with func to return mesh type (Mesh module)
    #     ione = one(integer)
    #     # @inbounds for i ∈ eachindex(nzval)
    #     #     nzval[i] = fzero
    #     # end
    #     kernel! = set_nzval(backend)
    #     kernel!(nzval, fzero, ndrange = length(nzval))
    #     cIndex = zero(integer) # replace with func to return mesh type (Mesh module)
    #     nIndex = zero(integer) # replace with func to return mesh type (Mesh module)
    #     @inbounds for cID ∈ eachindex(cells)
    #         cell = cells[cID]
    #         # (; facesID, nsign, neighbours) = cell
    #         # @inbounds for fi ∈ eachindex(facesID)
    #         @inbounds for fi ∈ cell.faces_range
    #             fID = cell_faces[fi]
    #             ns = cell_nsign[fi] # normal sign
    #             face = faces[fID]
    #             nID = cell_neighbours[fi]
    #             cellN = cells[nID]

    #             start = colptr[cID]
    #             offset = findfirst(isequal(cID),@view rowval[start:end]) - ione
    #             cIndex = start + offset

    #             start = colptr[nID]
    #             offset = findfirst(isequal(cID),@view rowval[start:end]) - ione
    #             nIndex = start + offset
    #             $(assignment_block_1...)    
    #         end
    #         b[cID] = fzero
    #         volume = cell.volume
    #         $(assignment_block_2...)
    #         $(assignment_block_3...)
    #        end

    # quote
        # Extract number of terms and sources
        nTerms = length(model.terms)
        nSources = length(model.sources)

        # Define variables for function
        mesh = model.terms[1].phi.mesh
        # precon = eqn.preconditioner

        # Deconstructors to get lower-level variables for function
        # (; A, b) = eqn.equation
        (; terms, sources) = model
        (; faces, cells, cell_faces, cell_neighbours, cell_nsign) = mesh
        backend = _get_backend(mesh)
        A_array = _A(eqn)
        b_array = _b(eqn)
        
        # Get types and set float(zero) and integer(one)
        integer = _get_int(mesh)
        float = _get_float(mesh)
        fzero = zero(float) # replace with func to return mesh type (Mesh module)
        ione = one(integer)

        # Deconstruct sparse array dependent on sparse arrays type
        rowval_array = _rowval(A_array)
        colptr_array = _colptr(A_array)
        nzval_array = _nzval(A_array)
        
        # Kernel to set nzval array to 0
        kernel! = set_nzval(backend)
        kernel!(nzval_array, fzero, ndrange = length(nzval_array))
        KernelAbstractions.synchronize(backend)
        # println(typeof(eqn))

        # Set initial values for indexing of nzval array
        cIndex = zero(integer) # replace with func to return mesh type (Mesh module)
        nIndex = zero(integer) # replace with func to return mesh type (Mesh module)
        offset = zero(integer)

        # Set b array to 0
        kernel! = set_b!(backend)
        kernel!(fzero, b_array, ndrange = length(b_array))
        KernelAbstractions.synchronize(backend)

        # Run schemes and sources calculations on all terms

        for i in 1:nTerms
            schemes_and_sources!(model.terms[i], 
                                nTerms, nSources, offset, fzero, ione, terms, rowval_array,
                                colptr_array, nzval_array, cIndex, nIndex, b_array,
                                faces, cells, cell_faces, cell_neighbours, cell_nsign, integer,
                                float, backend, runtime, prev)
            # KernelAbstractions.synchronize(backend)
        end

        # Free unneeded backend memory 
        nzval_array = nothing
        rowval_array = nothing
        colptr_array = nothing

        # Run sources calculations on all sources
        kernel! = sources!(backend)
        for i in 1:nSources
            (; field, sign) = sources[i]
            kernel!(field, sign, cells, b_array, ndrange = length(cells))
            # KernelAbstractions.synchronize(backend)
        end
        nothing
    # end
end


# function sparse_array_deconstructor(arr::SparseArrays.SparseMatrixCSC)
#     (; rowval, colptr, nzval) = arr
#     return rowval, colptr, nzval
# end

# function sparse_array_deconstructor(arr::CUDA.CUSPARSE.CuSparseMatrixCSC)
#     (; rowVal, colPtr, nzVal) = arr
#     return rowVal, colPtr, nzVal
# end

@kernel function set_nzval(nzval, fzero)
    i = @index(Global)

    @inbounds begin
        nzval[i] = fzero
    end
end

# function check_for_precon!(nzval, precon::Tuple, backend)
#     nothing
# end

# function check_for_precon!(nzval, precon, backend)
#     (; A) = precon
#     A_precon = A
#     copy_to_precon!(nzval, A_precon, backend)
# end

# function copy_to_precon!(nzval, A::SparseArrays.SparseMatrixCSC, backend)
#     nothing
# end

# function copy_to_precon!(nzval, A::CUDA.CUSPARSE.CuSparseMatrixCSC, backend)
#     rowval_precon, colptr_precon, nzval_precon = sparse_array_deconstructor(A)

#     kernel! = copy_to_precon_kernel!(backend)
#     kernel!(nzval, nzval_precon, ndrange = length(nzval))
# end

# @kernel function copy_to_precon_kernel!(nzval, nzval_precon)
#     i = @index(Global)

#     @inbounds begin
#         nzval_precon[i] = nzval[i]
#     end
# end

_nzval(A::CUDA.CUSPARSE.CuSparseMatrixCSC) = A.nzVal
_nzval(A::SparseArrays.SparseMatrixCSC) = A.nzval

_colptr(A::CUDA.CUSPARSE.CuSparseMatrixCSC) = A.colPtr
_colptr(A::SparseArrays.SparseMatrixCSC) = A.colptr

_rowval(A::CUDA.CUSPARSE.CuSparseMatrixCSC) = A.rowVal
_rowval(A::SparseArrays.SparseMatrixCSC) = A.rowval 
