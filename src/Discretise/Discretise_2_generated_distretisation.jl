export discretise!, update_equation!

function discretise!(
    eqn::ModelEquation{T,M,E,S,P}, prev, config) where {T<:VectorModel,M,E,S,P}
    (; hardware, runtime) = config
    (; backend, workgroup) = hardware

    # Retrieve variabels for defition
    mesh = eqn.model.terms[1].phi.mesh
    model = eqn.model

    # Sparse array and b accessor call
    A = _A(eqn)
    A0 = _A0(eqn)
    (; bx, by, bz) = eqn.equation

    # Sparse array fields accessors
    nzval = _nzval(A)
    nzval0 = _nzval(A0)
    colval = _colval(A)
    rowptr = _rowptr(A)


    # Call discretise kernel
    kernel! = _discretise_vector_model!(backend, workgroup)
    kernel!(model, model.terms, model.sources, mesh, nzval0, nzval, colval, rowptr, bx, by, bz, prev, runtime; ndrange = length(mesh.cells))
    KernelAbstractions.synchronize(backend)
end

@kernel function _discretise_vector_model!(
    model::Model{TN,SN,T,S}, terms, sources, mesh, nzval0::AbstractArray{F}, nzval, colval, rowptr, bx, by, bz, prev, runtime) where {TN,SN,T,S,F}
    i = @index(Global)
    # Extract mesh fields for kernel
    (; faces, cells, cell_faces, cell_neighbours, cell_nsign) = mesh

    @inbounds begin
        # Define workitem cell and extract required fields
        cell = cells[i]
        (; faces_range, volume) = cell


        # Set index for sparse array values on diagonal
        cIndex = spindex(rowptr, colval, i, i)

        # For loop over workitem cell faces
        ac_sum = zero(F)
        for fi in faces_range
            # Retrieve indices for discretisation
            fID = cell_faces[fi]
            ns = cell_nsign[fi] # normal sign
            face = faces[fID]
            nID = cell_neighbours[fi]
            cellN = cells[nID]
            
            # Set index for sparse array values at workitem cell neighbour index
            nIndex = spindex(rowptr, colval, i, nID)


            # Call scheme generated fucntion
            ac, an = _scheme!(model, terms, nzval0, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime)
            ac_sum += ac
            nzval0[nIndex] = an

        end

        
        # Call scheme source generated function NEEDS UPDATING!
        ac, bx1, by1, bz1 = _scheme_source!(model, terms, cell, i, cIndex, prev, runtime)
        
        nzval0[cIndex] = ac_sum + ac

        # Call sources generated function
        bx2, by2, bz2 = _sources!(model, sources, volume, i)
        bx[i] = bx1 + bx2
        by[i] = by1 + by2
        bz[i] = bz1 + bz2 
    end
end

function discretise!(
    eqn::ModelEquation{T,M,E,S,P}, prev, config) where {T<:ScalarModel,M,E,S,P}

    (; hardware, runtime) = config
    (; backend, workgroup) = hardware

    # Retrieve variabels for defition
    mesh = eqn.model.terms[1].phi.mesh
    model = eqn.model

    # Sparse array and b accessor call
    A = _A(eqn)
    b = _b(eqn)

    # Sparse array fields accessors
    nzval = _nzval(A)
    colval = _colval(A)
    rowptr = _rowptr(A)


    # Call discretise kernel
    kernel! = _discretise_scalar_model!(backend, workgroup)
    kernel!(model, model.terms, model.sources, mesh, nzval, colval, rowptr, b, prev, runtime; ndrange = length(mesh.cells))
    KernelAbstractions.synchronize(backend)
end

# Discretise kernel function
@kernel function _discretise_scalar_model!(
    model::Model{TN,SN,T,S}, terms, sources, mesh, nzval::AbstractArray{F}, colval, rowptr, b, prev, runtime) where {TN,SN,T,S,F}
    i = @index(Global)
    # Extract mesh fields for kernel
    (; faces, cells, cell_faces, cell_neighbours, cell_nsign) = mesh

    @inbounds begin
        # Define workitem cell and extract required fields
        cell = cells[i]
        (; faces_range, volume) = cell

        # Set index for sparse array values on diagonal!
        cIndex = spindex(rowptr, colval, i, i)

        # For loop over workitem cell faces
        ac_sum = zero(F)
        for fi in faces_range
            # Retrieve indices for discretisation
            fID = cell_faces[fi]
            ns = cell_nsign[fi] # normal sign
            face = faces[fID]
            nID = cell_neighbours[fi]
            cellN = cells[nID]
            
            # Set index for sparse array values at workitem cell neighbour index
            nIndex = spindex(rowptr, colval, i, nID)

            # Call scheme generated fucntion
            ac, an = _scheme!(model, terms, nzval, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime)
            ac_sum += ac
            nzval[nIndex] = an
        end
        
        # Call scheme source generated function
        ac, b1 = _scheme_source!(model, terms, cell, i, cIndex, prev, runtime)
        nzval[cIndex] = ac_sum + ac

        # Call sources generated function
        b2 = _sources!(model, sources, volume, i)
        b[i] = b2 + b1
    end
end

return_quote(x, t) = :(nothing)

# Scheme generated function definition
@generated function _scheme!(model::Model{TN,SN,T,S}, terms, nzval, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime) where {TN,SN,T,S}
    # Allocate expression array to store scheme function
    out = Expr(:block)

    # Loop over number of terms and store scheme function in array
    for t in 1:TN
        function_call_scheme = quote
            ac, an = scheme!(terms[$t], nzval, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime)
            AC += ac
            AN += an
        end
        push!(out.args, function_call_scheme)
    end
    # out
    quote
        AC = 0.0
        AN = 0.0
        $(out.args...)
        return AC, AN
    end
end

# Scheme source generated function definition
@generated function _scheme_source!(model::Model{TN,SN,T,S}, terms, cell, cID, cIndex, prev, runtime) where {TN,SN,T,S}
    # Allocate expression array to store scheme_source function
    out = Expr(:block)
    
    # Loop over number of terms and store scheme_source function in array
    if S.parameters[1].parameters[1] <: AbstractScalarField
        for t in 1:TN
            function_call_scheme_source = quote
                ac, b = scheme_source!(terms[$t], cell, cID, cIndex, prev, runtime)
                AC += ac
                B += b
            end
            push!(out.args, function_call_scheme_source)
        end
        return quote
            ac = 0.0
            b = 0.0
            AC = 0.0
            B = 0.0
            $(out.args...)
            return AC, B
        end
    elseif S.parameters[1].parameters[1] <: AbstractVectorField
        for t in 1:TN
            function_call_scheme_source = quote
                ac, bx = scheme_source!(terms[$t], cell, cID, cIndex, prev.x, runtime)
                ac, by = scheme_source!(terms[$t], cell, cID, cIndex, prev.y, runtime)
                ac, bz = scheme_source!(terms[$t], cell, cID, cIndex, prev.z, runtime)
                AC += ac # assuming ac's for all directions are equal
                BX += bx
                BY += by
                BZ += bz
            end
            push!(out.args, function_call_scheme_source)
        end
        return quote
            ac = 0.0
            bx = 0.0
            by = 0.0
            bz = 0.0
            AC = 0.0
            BX = 0.0
            BY = 0.0
            BZ = 0.0
            $(out.args...)
            return AC, BX, BY, BZ
        end
    end
end

# Sources generated function definition
@generated function _sources!(model::Model{TN,SN,T,S}, sources, volume, cID) where {TN,SN,T,S}
    # Allocate expression array to store source function
    out = Expr(:block)

    # Loop over number of terms and store source function in array
    if S.parameters[1].parameters[1] <: AbstractScalarField
        for s in 1:SN
            expression_call_sources = quote
                (; field, sign) = sources[$s]
                B += sign*field[cID]*volume
            end
            push!(out.args, expression_call_sources)
        end
        return quote
            B = 0.0
            $(out.args...)
            return B
        end
    elseif S.parameters[1].parameters[1] <: AbstractVectorField
        for s in 1:SN
            expression_call_sources = quote
                (; field, sign) = sources[$s]
                Bx += sign*field.x[cID]*volume
                By += sign*field.y[cID]*volume
                Bz += sign*field.z[cID]*volume
            end
            push!(out.args, expression_call_sources)
        end
        return quote
            Bx = 0.0
            By = 0.0
            Bz = 0.0
            $(out.args...)
            return Bx, By, Bz
        end
    end
end

@kernel function set_nzval!(nzval::AbstractArray{T}) where T
    i = @index(Global)

    @inbounds begin
        nzval[i] = zero(T)
    end
end

# Reset main equation to reuse in segregated solver
function update_equation!(eqn::ModelEquation{T,M,E,S,P}, config) where {T<:VectorModel,M,E,S,P}
    (; hardware, runtime) = config
    (; backend, workgroup) = hardware

    # Sparse array and b accessor call
    A = _A(eqn)
    A0 = _A0(eqn)

    # Sparse array fields accessors
    nzval0 = _nzval(A0)
    nzval = _nzval(A)

    # Call set nzval to zero kernel
    kernel! = _update_equation!(backend, workgroup)
    kernel!(nzval, nzval0, ndrange = length(nzval0))
    KernelAbstractions.synchronize(backend)
end

@kernel function _update_equation!(nzval, nzval0) 
    i = @index(Global)

    @inbounds begin
        nzval[i] = nzval0[i]
    end
end