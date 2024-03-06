export discretise!, nzval, rowval, colptr

discretise!(eqn, prev, runtime) = _discretise!(eqn.model, eqn, prev, runtime)

@generated function _discretise!(
    model::Model{TN,SN,T,S}, eqn, prev, runtime
    ) where {TN,SN,T,S}

    nTerms = TN
    nSources = SN

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

    quote
        # Extract number of terms and sources
        nTerms = TN
        nSources = SN

        # Define variables for function
        mesh = model.terms[1].phi.mesh
        precon = eqn.preconditioner

        # Deconstructors to get lower-level variables for function
        (; A, b) = eqn.equation
        # A = A
        # b = b
        (; terms, sources) = model
        (; faces, cells, cell_faces, cell_neighbours, cell_nsign) = mesh
        
        # Get types and create float(zero) and integer(one)
        integer = _get_int(mesh)
        float = _get_float(mesh)
        backend = _get_backend(mesh)
        fzero = zero(float) # replace with func to return mesh type (Mesh module)
        ione = one(integer)

        # (; faces, cells, ) = mesh
        # (; rowval, colptr, nzval) = A
        # Deconstruct sparse array dependent on sparse arrays type
        rowval_array = rowval(A)
        colptr_array = colptr(A)
        nzval_array = nzval(A)

        # @inbounds for i ∈ eachindex(nzval)
        #     nzval[i] = fzero
        # end
        
        # Kernel to set nzval array
        kernel! = set_nzval(backend)
        kernel!(nzval_array, fzero, ndrange = length(nzval_array))
        KernelAbstractions.synchronize(backend)

        # Set initial values for indexing of nzval array
        cIndex = zero(integer) # replace with func to return mesh type (Mesh module)
        nIndex = zero(integer) # replace with func to return mesh type (Mesh module)
        offset = zero(integer)

        # Assign storage for sources arrays
        sources_field = Array{typeof(sources[1].field)}(undef, length(sources))
        sources_sign = Array{typeof(sources[1].sign)}(undef, length(sources))

        for i in eachindex(sources)
            sources_field[i] = sources[i].field
            sources_sign[i] = sources[i].sign
        end

        # Move sources arrays to required backend
        sources_field = _convert_array!(sources_field, backend)
        sources_sign = _convert_array!(sources_sign, backend)

        # Set b array to 0
        kernel! = set_b!(backend)
        kernel!(fzero, b, ndrange = length(b))
        KernelAbstractions.synchronize(backend)

        # Run schemes and sources calculations on all terms
        for i in 1:nTerms
            schemes_and_sources!(model.terms[i], 
                                nTerms, nSources, offset, fzero, ione, terms, sources_field,
                                sources_sign, rowval_array, colptr_array, nzval_array, cIndex, nIndex,
                                b, faces, cells, cell_faces, cell_neighbours, cell_nsign, integer,
                                float, backend, runtime, prev)
            KernelAbstractions.synchronize(backend)
        end

        # Run sources calculations on all sources
        kernel! = sources!(backend)
        for i in 1:nSources
            (; field, sign) = sources[i]
            kernel!(field, sign, cells, b, ndrange = length(cells))
            KernelAbstractions.synchronize(backend)
        end

        # Copy nzval array to preconditioner if running on GPU and if preconditioner is not empty
        # check_for_precon!(nzval, precon, backend)
        # end
        nothing
    end
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

nzval(A::CUDA.CUSPARSE.CuSparseMatrixCSC) = A.nzVal
nzval(A::SparseArrays.SparseMatrixCSC) = A.nzval

colptr(A::CUDA.CUSPARSE.CuSparseMatrixCSC) = A.colPtr
colptr(A::SparseArrays.SparseMatrixCSC) = A.colptr

rowval(A::CUDA.CUSPARSE.CuSparseMatrixCSC) = A.rowVal
rowval(A::SparseArrays.SparseMatrixCSC) = A.rowval 