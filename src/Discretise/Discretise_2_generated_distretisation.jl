export discretise!

function discretise!(eqn, prev, runtime, nfaces, nbfaces)
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

    # kernel! = _discretise!(backend)
    # kernel!(model, model.terms, model.sources, mesh, nzval_array, rowval_array, colptr_array, b_array, prev, runtime, fzero, ione; ndrange = length(mesh.cells))
    # KernelAbstractions.synchronize(backend)

    # CUDA.@allowscalar nbfaces = mesh.boundaries[end].IDs_range[end]
    # nfaces = length(mesh.faces)
    internalfaces = nfaces - nbfaces

    kernel! = _discretise_face!(backend, 256, internalfaces)
    kernel!(model, model.terms, model.sources, mesh, nzval_array, rowval_array, colptr_array, b_array, prev, runtime, fzero, ione; ndrange = internalfaces)
    KernelAbstractions.synchronize(backend)

    kernel! = _discretise_face_sources!(backend)
    kernel!(model, model.terms, model.sources, mesh, nzval_array, rowval_array, colptr_array, b_array, prev, runtime, fzero, ione; ndrange = length(mesh.cells))
    KernelAbstractions.synchronize(backend)
end

function spindex(rowval, colptr, i, j)
    # view_range = @view rowval[colptr[j]:colptr[j+1]]
    # findnext(isequal(i), view_range, i) + colptr[j]

    start_ind = colptr[j]
    end_ind = colptr[j+1]

    ind = zero(typeof(start_ind))
    for nzi in start_ind:end_ind
        if rowval[nzi] == i
            ind = nzi
            break
        end
    end
    return ind
end

@kernel function _discretise_face!(
    model::Model{TN,SN,T,S}, terms, sources, mesh, nzval_array, @Const(rowval_array), @Const(colptr_array), b_array, prev, @Const(runtime), fzero, ione) where {TN,SN,T,S}
    ic = @index(Global,Linear)
    li = @index(Local,Linear)


    N = @groupsize()[1]
    test = @localmem Int32 N


    @uniform nbfaces = mesh.boundaries[end].IDs_range[end]
    fID = ic + nbfaces
    # backend = _get_backend(mesh)
    @uniform (; faces, cells, cell_faces, cell_neighbours, cell_nsign) = mesh
  
    
    face = faces[fID]
    owners = face.ownerCells
    cID1 = owners[1]
    cID2 = owners[2]
    cell1 = cells[cID1]
    cell2 = cells[cID2]

    # cIndex1 = nzval_index(colptr_array, rowval_array, cID1, cID1, ione)
    # cIndex2 = nzval_index(colptr_array, rowval_array, cID2, cID2, ione)

    # nIndex1 = nzval_index(colptr_array, rowval_array, cID2, cID1, ione)
    # nIndex2 = nzval_index(colptr_array, rowval_array, cID1, cID2, ione)

    cIndex1 = spindex(rowval_array, colptr_array, cID1, cID1)
    cIndex2 = spindex(rowval_array, colptr_array,cID2, cID2)

    nIndex1 = spindex(rowval_array, colptr_array, cID1, cID2)
    nIndex2 = spindex(rowval_array, colptr_array, cID2, cID1)

    # _scheme!(model, terms, nzval_array, cell1, face,  cell2, ione, cIndex1, nIndex1, fID, prev, runtime)
    # _scheme!(model, terms, nzval_array, cell2, face,  cell1, -ione, cIndex2, nIndex2, fID, prev, runtime)

    _scheme!(model, terms, nzval_array, face, cell1,  cell2, ione, cIndex1, nIndex1, cIndex2, nIndex2, fID, prev, runtime)

    # @inbounds begin
        # (; faces_range, volume) = cell

        # cIndex = nzval_index(colptr_array, rowval_array, i, i, ione)

        # for fi in faces_range
        #     fID = cell_faces[fi]
        #     ns = cell_nsign[fi] # normal sign
        #     face = faces[fID]
        #     nID = cell_neighbours[fi]
        #     cellN = cells[nID]

            # nIndex = nzval_index(colptr_array, rowval_array, nID, i, ione)

        #     _scheme!(model, terms, nzval_array, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime)

        # end
    # end
end

@kernel function _discretise_face_sources!(
    model::Model{TN,SN,T,S}, terms, sources, mesh, nzval_array, @Const(rowval_array), @Const(colptr_array), b_array, prev, runtime, fzero, ione) where {TN,SN,T,S}
    i = @index(Global)
    
    @uniform (; faces, cells, cell_faces, cell_neighbours, cell_nsign) = mesh

    # @inbounds begin
    cell = cells[i]
    (; volume) = cell

    # cIndex = nzval_index(colptr_array, rowval_array, i, i, ione)
    cIndex = spindex(rowval_array, colptr_array, i, i)

    _scheme_source!(model, terms, b_array, nzval_array, cell, i, cIndex, prev, runtime)
    _sources!(model, sources, b_array, volume, i)
    # end
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

@generated function _scheme!(model::Model{TN,SN,T,S}, terms, nzval_array, face, cell1,   cell2, ns, cIndex1, nIndex1, cIndex2, nIndex2, fID, prev, runtime) where {TN,SN,T,S}
    
    # # Implementation 2
    out = Expr(:block)
    for t in 1:TN
        function_call_scheme = quote
            # scheme!(model.terms[$t], nzval_array, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime)
            scheme!(terms[$t], nzval_array, cell1, face,  cell2, ns, cIndex1, nIndex1, fID, prev, runtime)
            scheme!(terms[$t], nzval_array, cell2, face,  cell1, -ns, cIndex2, nIndex2, fID, prev, runtime)
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