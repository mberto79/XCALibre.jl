export discretise!

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

    kernel!(model, model.terms, model.sources, mesh, nzval_array, rowval_array, colptr_array, b_array, prev, runtime, fzero, ione; ndrange = length(mesh.cells))
    KernelAbstractions.synchronize(backend)
end

@kernel function _discretise!(
    model::Model{TN,SN,T,S}, terms, sources, mesh, nzval_array, rowval_array, colptr_array, b_array, prev, runtime, fzero, ione) where {TN,SN,T,S}
    i = @index(Global)
    
    (; faces, cells, cell_faces, cell_neighbours, cell_nsign) = mesh

    # @inbounds begin
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

            _scheme!(model, terms, nzval_array, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime)

        end


        _scheme_source!(model, terms, b_array, nzval_array, cell, i, cIndex, prev, runtime)
        _sources!(model, sources, b_array, volume, i)
    # end
end

return_quote(x, t) = :(nothing)

@generated function _scheme!(model::Model{TN,SN,T,S}, terms, nzval_array, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime) where {TN,SN,T,S}
    
    # # Implementation 2
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
    
    # Implementation 2
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
   
    # Implementation 2
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