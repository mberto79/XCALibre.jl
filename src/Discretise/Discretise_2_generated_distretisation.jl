export discretise!

# Discretise Function
function discretise!(eqn, prev, config)

    # backend = _get_backend(mesh)
    (; hardware, runtime) = config
    (; backend, workgroup) = hardware

    # Retrieve variabels for defition
    mesh = eqn.model.terms[1].phi.mesh
    model = eqn.model
    integer = _get_int(mesh)
    float = _get_float(mesh)
    fzero = zero(float)
    ione = one(integer)

    # Sparse array and b accessor call
    A_array = _A(eqn)
    b_array = _b(eqn)

    # Sparse array fields accessors
    nzval_array = _nzval(A_array)
    rowval_array = _rowval(A_array)
    colptr_array = _colptr(A_array)

    # Call set nzval to zero kernel
    kernel! = set_nzval!(backend, workgroup)
    kernel!(nzval_array, fzero, ndrange = length(nzval_array))
    KernelAbstractions.synchronize(backend)

    # Call set b to zero kernel
    kernel! = set_b!(backend, workgroup)
    kernel!(b_array, fzero, ndrange = length(b_array))
    KernelAbstractions.synchronize(backend)

    # Call discretise kernel
    kernel! = _discretise!(backend, workgroup)
    kernel!(model, model.terms, model.sources, mesh, nzval_array, rowval_array, colptr_array, b_array, prev, runtime, fzero, ione; ndrange = length(mesh.cells))
    KernelAbstractions.synchronize(backend)
end

# Discretise kernel function
@kernel function _discretise!(
    model::Model{TN,SN,T,S}, terms, sources, mesh, nzval_array, rowval_array, colptr_array, b_array, prev, runtime, fzero, ione) where {TN,SN,T,S}
    i = @index(Global)
    
    # Extract mesh fields for kernel
    (; faces, cells, cell_faces, cell_neighbours, cell_nsign) = mesh

    @inbounds begin
        # Define workitem cell and extract required fields
        cell = cells[i]
        (; faces_range, volume) = cell

        # Set index for sparse array values on diagonal
        cIndex = nzval_index(colptr_array, rowval_array, i, i, ione)

        # For loop over workitem cell faces
        for fi in faces_range
            # Retrieve indices for discretisation
            fID = cell_faces[fi]
            ns = cell_nsign[fi] # normal sign
            face = faces[fID]
            nID = cell_neighbours[fi]
            cellN = cells[nID]
            
            # Set index for sparse array values at workitem cell neighbour index
            nIndex = nzval_index(colptr_array, rowval_array, nID, i, ione)

            # Call scheme generated fucntion
            _scheme!(model, terms, nzval_array, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime)
        end
        # Call scheme source generated function
        _scheme_source!(model, terms, b_array, nzval_array, cell, i, cIndex, prev, runtime)

        # Call sources generated function
        _sources!(model, sources, b_array, volume, i)
    end
end

return_quote(x, t) = :(nothing)

# Scheme generated function definition
@generated function _scheme!(model::Model{TN,SN,T,S}, terms, nzval_array, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime) where {TN,SN,T,S}
    # Allocate expression array to store scheme function
    out = Expr(:block)

    # Loop over number of terms and store scheme function in array
    for t in 1:TN
        function_call_scheme = quote
            scheme!(terms[$t], nzval_array, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime)
        end
        push!(out.args, function_call_scheme)
    end
    out
end

# Scheme source generated function definition
@generated function _scheme_source!(model::Model{TN,SN,T,S}, terms, b, nzval_array, cell, cID, cIndex, prev, runtime) where {TN,SN,T,S}[]
    # Allocate expression array to store scheme_source function
    out = Expr(:block)
    
    # Loop over number of terms and store scheme_source function in array
    for t in 1:TN
        function_call_scheme_source = quote
            scheme_source!(terms[$t], b, nzval_array, cell, cID, cIndex, prev, runtime)
        end
        push!(out.args, function_call_scheme_source)
    end
    out
end

# Sources generated function definition
@generated function _sources!(model::Model{TN,SN,T,S}, sources, b, volume, cID) where {TN,SN,T,S}
    # Allocate expression array to store source function
    out = Expr(:block)

    # Loop over number of terms and store source function in array
    for s in 1:SN
        expression_call_sources = quote
            (; field, sign) = sources[$s]
            Atomix.@atomic b[cID] += sign*field[cID]*volume
        end
        push!(out.args, expression_call_sources)
    end
    out
end

# Set nzval array to zero kernel
@kernel function set_nzval!(nzval, fzero)
    i = @index(Global)

    @inbounds begin
        nzval[i] = fzero
    end
end

# Set b array to zero kernel
@kernel function set_b!(b, fzero)
    i = @index(Global)

    @inbounds begin
        b[i] = fzero
    end
end