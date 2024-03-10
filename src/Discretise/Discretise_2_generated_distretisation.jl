export discretise!

# discretise!(eqn, prev, runtime) = _discretise!(eqn.model, eqn, prev, runtime)

# function _discretise!(
#     model::Model{TN,SN,T,S}, eqn, prev, runtime
#     ) where {TN,SN,T,S}

#     # nTerms = TN
#     # nSources = SN

#     # assignment_block_1 = Expr[] # Ap
#     # assignment_block_2 = Expr[] # An or b
#     # assignment_block_3 = Expr[] # b (sources)

#     # # Loop for operators
#     # for t ∈ 1:nTerms
#     #     function_call = quote
#     #         scheme!(
#     #             model.terms[$t], nzval, cell, face, cellN, ns, cIndex, nIndex, fID, prev, runtime
#     #             )
#     #     end
#     #     push!(assignment_block_1, function_call)

#     #     assign_source = quote
#     #         scheme_source!(model.terms[$t], b, nzval, cell, cID, cIndex, prev, runtime)
#     #     end
#     #     push!(assignment_block_2, assign_source)
#     # end

#     # # Loop for sources
#     # for s ∈ 1:nSources
#     #     add_source = quote
#     #         (; field, sign) = model.sources[$s]
#     #         b[cID] += sign*field[cID]*volume
#     #     end
#     #     push!(assignment_block_3, add_source)
#     # end

#     # quote
#     #     (; A, b) = eqn.equation
#     #     mesh = model.terms[1].phi.mesh
#     #     integer = _get_int(mesh)
#     #     float = _get_float(mesh)
#     #     backend = _get_backend(mesh)
#     #     # (; faces, cells, ) = mesh
#     #     (; faces, cells, cell_faces, cell_neighbours, cell_nsign) = mesh
#     #     # (; rowval, colptr, nzval) = A
#     #     rowval, colptr, nzval = sparse_array_deconstructor(A)
#     #     fzero = zero(float) # replace with func to return mesh type (Mesh module)
#     #     ione = one(integer)
#     #     # @inbounds for i ∈ eachindex(nzval)
#     #     #     nzval[i] = fzero
#     #     # end
#     #     kernel! = set_nzval(backend)
#     #     kernel!(nzval, fzero, ndrange = length(nzval))
#     #     cIndex = zero(integer) # replace with func to return mesh type (Mesh module)
#     #     nIndex = zero(integer) # replace with func to return mesh type (Mesh module)
#     #     @inbounds for cID ∈ eachindex(cells)
#     #         cell = cells[cID]
#     #         # (; facesID, nsign, neighbours) = cell
#     #         # @inbounds for fi ∈ eachindex(facesID)
#     #         @inbounds for fi ∈ cell.faces_range
#     #             fID = cell_faces[fi]
#     #             ns = cell_nsign[fi] # normal sign
#     #             face = faces[fID]
#     #             nID = cell_neighbours[fi]
#     #             cellN = cells[nID]

#     #             start = colptr[cID]
#     #             offset = findfirst(isequal(cID),@view rowval[start:end]) - ione
#     #             cIndex = start + offset

#     #             start = colptr[nID]
#     #             offset = findfirst(isequal(cID),@view rowval[start:end]) - ione
#     #             nIndex = start + offset
#     #             $(assignment_block_1...)    
#     #         end
#     #         b[cID] = fzero
#     #         volume = cell.volume
#     #         $(assignment_block_2...)
#     #         $(assignment_block_3...)
#     #        end

#     # quote
#         # Extract number of terms and sources
#         # nTerms = length(model.terms)
#         # nSources = length(model.sources)

#         # # Define variables for function
#         # mesh = model.terms[1].phi.mesh
#         # # precon = eqn.preconditioner

#         # # Deconstructors to get lower-level variables for function
#         # # (; A, b) = eqn.equation
#         # (; terms, sources) = model
#         # (; faces, cells, cell_faces, cell_neighbours, cell_nsign) = mesh
#         # backend = _get_backend(mesh)
#         # A_array = _A(eqn)
#         # b_array = _b(eqn)
        
#         # # Get types and set float(zero) and integer(one)
#         # integer = _get_int(mesh)
#         # float = _get_float(mesh)
#         # fzero = zero(float) # replace with func to return mesh type (Mesh module)
#         # ione = one(integer)

#         # # Deconstruct sparse array dependent on sparse arrays type
#         # rowval_array = _rowval(A_array)
#         # colptr_array = _colptr(A_array)
#         # nzval_array = _nzval(A_array)
        
#         # # Kernel to set nzval array to 0
#         # kernel! = set_nzval(backend)
#         # kernel!(nzval_array, fzero, ndrange = length(nzval_array))
#         # KernelAbstractions.synchronize(backend)
#         # # println(typeof(eqn))

#         # # Set initial values for indexing of nzval array
#         # cIndex = zero(integer) # replace with func to return mesh type (Mesh module)
#         # nIndex = zero(integer) # replace with func to return mesh type (Mesh module)
#         # offset = zero(integer)

#         # # Set b array to 0
#         # kernel! = set_b!(backend)
#         # kernel!(fzero, b_array, ndrange = length(b_array))
#         # KernelAbstractions.synchronize(backend)

#         # # Run schemes and sources calculations on all terms

#         # for i in 1:nTerms
#         #     schemes_and_sources!(model.terms[i], 
#         #                         nTerms, nSources, offset, fzero, ione, terms, rowval_array,
#         #                         colptr_array, nzval_array, cIndex, nIndex, b_array,
#         #                         faces, cells, cell_faces, cell_neighbours, cell_nsign, integer,
#         #                         float, backend, runtime, prev)
#         #     # KernelAbstractions.synchronize(backend)
#         # end

#         # # Free unneeded backend memory 
#         # nzval_array = nothing
#         # rowval_array = nothing
#         # colptr_array = nothing

#         # # Run sources calculations on all sources
#         # kernel! = sources!(backend)
#         # for i in 1:nSources
#         #     (; field, sign) = sources[i]
#         #     kernel!(field, sign, cells, b_array, ndrange = length(cells))
#         #     # KernelAbstractions.synchronize(backend)
#         # end
#         # nothing
#     # end
# end

function discretise!(eqn, prev, runtime)
    mesh = eqn.model.terms[1].phi.mesh
    model = eqn.model

    backend = _get_backend(mesh)
    integer = _get_int(mesh)
    float = _get_float(mesh)
    fzero = zero(float)
    ione = one(integer)
    # cIndex = zero(integer)
    # nIndex = zero(inetger)

    A_array = _A(eqn)
    b_array = _b(eqn)

    nzval_array = _nzval(A_array)
    rowval_array = _rowval(A_array)
    colptr_array = _colptr(A_array)

    kernel! = set_nzval!(backend)
    kernel!(nzval_array, fzero, ndrange = length(nzval_array))
    KernelAbstractions.synchronize(backend)

    kernel! = set_b!(backend)
    kernel!(b_array, fzero, ndrange = length(b_array))
    KernelAbstractions.synchronize(backend)

    kernel! = _discretise!(backend)
    kernel!(model, mesh, nzval_array, rowval_array, colptr_array, b_array, prev, runtime, fzero, ione, _scheme!, _scheme_source!, _sources!, ndrange = length(mesh.cells))
    KernelAbstractions.synchronize(backend)
end

@kernel function _discretise!(
    model::Model{TN,SN,T,S}, mesh, nzval_array, rowval_array, colptr_array, b_array, prev, runtime, fzero, ione, gfunc_1::F1, gfunc_2::F2, gfunc_3::F3) where {TN,SN,T,S,F1,F2,F3}
    i = @index(Global)
    
    (; faces, cells, cell_faces, cell_neighbours, cell_nsign) = mesh
    (; terms, sources) = model

    @inbounds begin
        cell = cells[i]
        (; faces_range, volume) = cell

        cIndex = nzval_index(colptr_array, rowval_array, i, i, ione)

        for fi in faces_range
            fID = cell_faces[fi]
            ns = cell_nsign[fi] # normal sign
            face = faces[fID]
            nID = cell_neighbours[fi]
            cellN = cells[nID]

            nIndex = nzval_index(colptr_array, rowval_array, nID, i, ione)

            gfunc_1(model, terms, nzval_array, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime)

            # scheme!(terms[1], nzval_array, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime)
            # scheme!(term2, nzval_array, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime)
            # scheme!(term3, nzval_array, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime)

        end
        # b_array[i] = fzero
        gfunc_2(model, terms, b_array, nzval_array, cell, i, cIndex, prev, runtime)
        gfunc_3(model, sources, b_array, volume, i)
    end
end

@generated function _scheme!(model::Model{TN,SN,T,S}, terms, nzval_array, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime) where {TN,SN,T,S}
    
    # nTerms = TN
    # assignment_block = Expr[]
    # for t in 1:nTerms
    #     function_call_scheme = quote
    #         # scheme!(model.terms[$t], nzval_array, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime)
    #         scheme!(terms[$t], nzval_array, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime)
    #     end
    #     push!(assignment_block, function_call_scheme)
    # end
    # quote
    #     $(assignment_block...)
    # end

    out = Expr(:block)
    
    for t in 1:TN
        function_call_scheme = quote
            # scheme!(model.terms[$t], nzval_array, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime)
            scheme!(terms[$t], nzval_array, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime)
        end
        push!(out.args, function_call_scheme)
    end
    out
end

@generated function _scheme_source!(model::Model{TN,SN,T,S}, terms, b, nzval_array, cell, cID, cIndex, prev, runtime) where {TN,SN,T,S}
    # nTerms = TN
    # assign_source = Expr[]
    # for t in 1:nTerms
    #     function_call_scheme_source = quote
    #         # scheme_source!(model.terms[$t], b, nzval_array, cell, cID, cIndex, prev, runtime)
    #         scheme_source!(terms[$t], b, nzval_array, cell, cID, cIndex, prev, runtime)
    #     end
    #     push!(assign_source, function_call_scheme_source)
    # end
    # quote
    #     $(assign_source...)
    # end
    out = Expr(:block)
    for t in 1:TN
        function_call_scheme_source = quote
            # scheme_source!(model.terms[$t], b, nzval_array, cell, cID, cIndex, prev, runtime)
            scheme_source!(terms[$t], b, nzval_array, cell, cID, cIndex, prev, runtime)
        end
        push!(out.args, function_call_scheme_source)
    end
    out
end

@generated function _sources!(model::Model{TN,SN,T,S}, sources, b, volume, cID) where {TN,SN,T,S}
    # nSources = SN
    # add_source = Expr[]
    # for s in 1:nSources
    #     expression_call_sources = quote
    #         # (; field, sign) = model.sources[$s]
    #         (; field, sign) = sources[$s]
    #         b[cID] += sign*field[cID]*volume
    #     end
    #     push!(add_source, expression_call_sources)
    # end
    # quote
    #     $(add_source...)
    # end

    out = Expr(:block)
    for s in 1:SN
        expression_call_sources = quote
            # (; field, sign) = model.sources[$s]
            (; field, sign) = sources[$s]
            Atomix.@atomic b[cID] += sign*field[cID]*volume
        end
        push!(out.args, expression_call_sources)
    end
    out
end

@kernel function set_nzval!(nzval, fzero)
    i = @index(Global)

    @inbounds begin
        nzval[i] = fzero
    end
end

@kernel function set_b!(b, fzero)
    i = @index(Global)

    @inbounds begin
        # @synchronize
        b[i] = fzero
    end
end