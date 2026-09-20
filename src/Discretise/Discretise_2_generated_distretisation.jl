export discretise!, update_equation!

function discretise!(
    eqn::ModelEquation{T,M,E,S,P}, prev, config; rho_prev=eqn.model.terms[1].flux) where {T<:VectorModel,M,E,S,P}
    (; hardware, runtime) = config
    (; backend, workgroup, assembly) = hardware

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
    (; diag_nz, face_nz, owner_nz, neig_nz, gDiff) = eqn.equation

    # reset storage of sparse matrix
    z = zero(eltype(nzval))
    xcal_foreach(nzval, config) do i
        nzval0[i] = z 
    end

    _assemble_vector!(assembly, model, mesh, nzval0, diag_nz, face_nz, owner_nz, neig_nz,
        gDiff, bx, by, bz, prev, runtime, rho_prev, backend, workgroup)
end

function _assemble_vector!(::CellAssembly, model, mesh, nzval0, diag_nz, face_nz, owner_nz,
    neig_nz, gDiff, bx, by, bz, prev, runtime, rho_prev, backend, workgroup)

    ndrange = length(mesh.cells)
    kernel! = _discretise_vector_model!(_setup(backend, workgroup, ndrange)...)
    kernel!(model, model.terms, model.sources, mesh, nzval0, diag_nz, face_nz, gDiff,
        bx, by, bz, prev, runtime, rho_prev)
end

# Face pass writes both off-diagonals outright and accumulates the two diagonals atomically;
# the cell pass then adds the source contribution to the diagonal it already holds.
function _assemble_vector!(::FaceAssembly, model, mesh, nzval0, diag_nz, face_nz, owner_nz,
    neig_nz, gDiff, bx, by, bz, prev, runtime, rho_prev, backend, workgroup)

    nbfaces = length(mesh.boundary_cellsID)
    ndrange = length(mesh.faces) - nbfaces
    kernel! = _discretise_faces!(_setup(backend, workgroup, ndrange)...)
    kernel!(model, model.terms, mesh, nzval0, diag_nz, owner_nz, neig_nz, gDiff, nbfaces,
        prev, runtime)

    ndrange = length(mesh.cells)
    kernel! = _discretise_vector_cells!(_setup(backend, workgroup, ndrange)...)
    kernel!(model, model.terms, model.sources, mesh, nzval0, diag_nz, bx, by, bz, prev,
        runtime, rho_prev)
end

# Shared by the scalar and vector face passes: only the matrix depends on the face loop, and
# scheme! reads neither the cell struct nor the solution vector.
@kernel function _discretise_faces!(
    model::Model{TN,SN,T,S}, terms::TERMS, mesh, nzval::AbstractArray{F}, diag_nz, owner_nz,
    neig_nz, gDiff, nbfaces, prev, runtime) where {TN,SN,T,S,F,TERMS}
    i = @index(Global)

    @inbounds begin
        fID = i + nbfaces # internal faces follow the boundary faces
        face = mesh.faces[fID]
        ownerCells = face.ownerCells
        cP = ownerCells[1]
        cN = ownerCells[2]
        pIndex = diag_nz[cP]
        nIndex = diag_nz[cN]
        onz = owner_nz[fID]
        nnz = neig_nz[fID]
        gDiff_f = gDiff[fID]
        ns = one(eltype(mesh.cell_nsign))

        # Same two calls the cell loop makes from either side of the face, with the face read
        # once instead of twice. nothing stands in for the cell struct: no scheme! uses it.
        acP, anP = _scheme!(model, terms, nzval, nothing, face, gDiff_f, cN, ns, pIndex, onz, fID, prev, runtime)
        acN, anN = _scheme!(model, terms, nzval, nothing, face, gDiff_f, cP, -ns, nIndex, nnz, fID, prev, runtime)

        nzval[onz] = anP
        nzval[nnz] = anN
        Atomix.@atomic nzval[pIndex] += acP
        Atomix.@atomic nzval[nIndex] += acN
    end
end

@kernel function _discretise_vector_cells!(
    model::Model{TN,SN,T,S}, terms::TERMS, sources::SRCS, mesh, nzval::AbstractArray{F},
    diag_nz, bx, by, bz, prev, runtime, rho_prev) where {TN,SN,T,S,F,TERMS,SRCS}
    i = @index(Global)

    @inbounds begin
        cell = mesh.cells[i]
        cIndex = diag_nz[i]
        ac, bx1, by1, bz1 = _scheme_source!(model, terms, cell, i, cIndex, prev, runtime, rho_prev)
        nzval[cIndex] += ac # face pass already accumulated the flux contributions
        bx2, by2, bz2 = _sources!(model, sources, cell.volume, i)
        bx[i] = bx1 + bx2
        by[i] = by1 + by2
        bz[i] = bz1 + bz2
    end
end

# @kernel function _discretise_vector_model!(
#     model::Model{TN,SN,T,S}, terms, sources, mesh, nzval0::AbstractArray{F}, nzval, colval, rowptr, bx, by, bz, prev, runtime) where {TN,SN,T,S,F}
@kernel function _discretise_vector_model!(
    model::Model{TN,SN,T,S}, terms::TERMS, sources::SRCS, mesh, nzval0::AbstractArray{F}, diag_nz, face_nz, gDiff, bx, by, bz, prev, runtime, rho_prev) where {TN,SN,T,S,F,TERMS,SRCS}
    i = @index(Global)
    # Extract mesh fields for kernel
    (; faces, cells, cell_faces, cell_neighbours, cell_nsign) = mesh

    @inbounds begin
        # Define workitem cell and extract required fields
        cell = cells[i]
        (; faces_range, volume) = cell


        cIndex = diag_nz[i]

        # For loop over workitem cell faces
        ac_sum = zero(F)
        for fi in faces_range
            # Retrieve indices for discretisation
            fID = cell_faces[fi]
            ns = cell_nsign[fi] # normal sign
            face = faces[fID]
            nID = cell_neighbours[fi]
            nIndex = face_nz[fi]


            # Call scheme generated fucntion
            ac, an = _scheme!(model, terms, nzval0, cell, face, gDiff[fID], nID, ns, cIndex, nIndex, fID, prev, runtime)
            ac_sum += ac
            nzval0[nIndex] = an

        end

        
        # Call scheme source generated function NEEDS UPDATING!
        ac, bx1, by1, bz1 = _scheme_source!(model, terms, cell, i, cIndex, prev, runtime, rho_prev)
        
        nzval0[cIndex] = ac_sum + ac

        # Call sources generated function
        bx2, by2, bz2 = _sources!(model, sources, volume, i)
        bx[i] = bx1 + bx2
        by[i] = by1 + by2
        bz[i] = bz1 + bz2 
    end
end

function discretise!(
    eqn::ModelEquation{T,M,E,S,P}, prev, config; rho_prev=eqn.model.terms[1].flux) where {T<:ScalarModel,M,E,S,P}

    (; hardware, runtime) = config
    (; backend, workgroup, assembly) = hardware

    # Retrieve variabels for defition
    mesh = eqn.model.terms[1].phi.mesh
    model = eqn.model

    # Sparse array and b accessor call
    A = _A(eqn)
    b = _b(eqn)

    # Sparse array fields accessors
    nzval = _nzval(A)
    (; diag_nz, face_nz, owner_nz, neig_nz, gDiff) = eqn.equation

    # reset storage of sparse matrix
    z = zero(eltype(nzval))
    xcal_foreach(nzval, config) do i
        nzval[i] = z 
    end

    _assemble_scalar!(assembly, model, mesh, nzval, diag_nz, face_nz, owner_nz, neig_nz,
        gDiff, b, prev, runtime, rho_prev, backend, workgroup)
end

function _assemble_scalar!(::CellAssembly, model, mesh, nzval, diag_nz, face_nz, owner_nz,
    neig_nz, gDiff, b, prev, runtime, rho_prev, backend, workgroup)

    ndrange = length(mesh.cells)
    kernel! = _discretise_scalar_model!(_setup(backend, workgroup, ndrange)...)
    kernel!(model, model.terms, model.sources, mesh, nzval, diag_nz, face_nz, gDiff, b,
        prev, runtime, rho_prev)
end

function _assemble_scalar!(::FaceAssembly, model, mesh, nzval, diag_nz, face_nz, owner_nz,
    neig_nz, gDiff, b, prev, runtime, rho_prev, backend, workgroup)

    nbfaces = length(mesh.boundary_cellsID)
    ndrange = length(mesh.faces) - nbfaces
    kernel! = _discretise_faces!(_setup(backend, workgroup, ndrange)...)
    kernel!(model, model.terms, mesh, nzval, diag_nz, owner_nz, neig_nz, gDiff, nbfaces,
        prev, runtime)

    ndrange = length(mesh.cells)
    kernel! = _discretise_scalar_cells!(_setup(backend, workgroup, ndrange)...)
    kernel!(model, model.terms, model.sources, mesh, nzval, diag_nz, b, prev, runtime, rho_prev)
end

@kernel function _discretise_scalar_cells!(
    model::Model{TN,SN,T,S}, terms::TERMS, sources::SRCS, mesh, nzval::AbstractArray{F},
    diag_nz, b, prev, runtime, rho_prev) where {TN,SN,T,S,F,TERMS,SRCS}
    i = @index(Global)

    @inbounds begin
        cell = mesh.cells[i]
        cIndex = diag_nz[i]
        ac, b1 = _scheme_source!(model, terms, cell, i, cIndex, prev, runtime, rho_prev)
        nzval[cIndex] += ac # face pass already accumulated the flux contributions
        b2 = _sources!(model, sources, cell.volume, i)
        b[i] = b1 + b2
    end
end

# Discretise kernel function
# @kernel function _discretise_scalar_model!(
#     model::Model{TN,SN,T,S}, terms, sources, mesh, nzval::AbstractArray{F}, colval, rowptr, b, prev, runtime) where {TN,SN,T,S,F}
@kernel function _discretise_scalar_model!(
    model::Model{TN,SN,T,S}, terms::TERMS, sources::SRCS, mesh, nzval::AbstractArray{F}, diag_nz, face_nz, gDiff, b, prev, runtime, rho_prev) where {TN,SN,T,S,F,TERMS,SRCS}

    i = @index(Global)
    # Extract mesh fields for kernel
    (; faces, cells, cell_faces, cell_neighbours, cell_nsign) = mesh

    @inbounds begin
        # Define workitem cell and extract required fields
        cell = cells[i]
        (; faces_range, volume) = cell

        cIndex = diag_nz[i]

        # For loop over workitem cell faces
        ac_sum = zero(F)
        for fi in faces_range
            # Retrieve indices for discretisation
            fID = cell_faces[fi]
            ns = cell_nsign[fi] # normal sign
            face = faces[fID]
            nID = cell_neighbours[fi]
            nIndex = face_nz[fi]

            # Call scheme generated fucntion
            ac, an = _scheme!(model, terms, nzval, cell, face, gDiff[fID], nID, ns, cIndex, nIndex, fID, prev, runtime)
            ac_sum += ac
            nzval[nIndex] = an
        end
        
        # Call scheme source generated function
        ac, b1 = _scheme_source!(model, terms, cell, i, cIndex, prev, runtime, rho_prev)
        nzval[cIndex] = ac_sum + ac

        # Call sources generated function
        b2 = _sources!(model, sources, volume, i)
        b[i] = b2 + b1
    end
end

return_quote(x, t) = :(nothing)

# Scheme generated function definition
# @generated function _scheme!(model::Model{TN,SN,T,S}, terms, nzval, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime) where {TN,SN,T,S}
@generated function _scheme!(
    model::Model{TN,SN,T,S}, terms::TERMS, nzval::AbstractArray{F}, cell, face,
    gDiff_f, nID, ns, cIndex, nIndex, fID, prev, runtime
    ) where {TN,SN,T,S,TERMS,F}
    # Allocate expression array to store scheme function
    out = Expr(:block)

    # Loop over number of terms and store scheme function in array
    for t in 1:TN
        function_call_scheme = quote
            ac, an = scheme!(terms[$t], nzval, cell, face, gDiff_f, nID, ns, cIndex, nIndex, fID, prev, runtime)
            AC += F(ac)
            AN += F(an)
        end
        push!(out.args, function_call_scheme)
    end
    # out
    quote
        z = zero(F)
        AC = z
        AN = z
        $(out.args...)
        return AC, AN
    end
end

# Scheme source generated function definition
@generated function _scheme_source!(model::Model{TN,SN,T,S}, terms::TERMS, cell::Cell{F}, cID, cIndex, prev, runtime, rho_prev) where {TN,SN,T,S,TERMS,F}
    # Allocate expression array to store scheme_source function
    out = Expr(:block)
    
    # Loop over number of terms and store scheme_source function in array
    if S.parameters[1].parameters[1] <: AbstractScalarField
        for t in 1:TN
            function_call_scheme_source = quote
                ac, b = scheme_source!(terms[$t], cell, cID, cIndex, prev, runtime, rho_prev)
                AC += F(ac)
                B += F(b)
            end
            push!(out.args, function_call_scheme_source)
        end
        return quote
            z = zero(F)
            ac = z
            b = z
            AC = z
            B = z
            $(out.args...)
            return AC, B
        end
    elseif S.parameters[1].parameters[1] <: AbstractVectorField
        for t in 1:TN
            function_call_scheme_source = quote
                ac, bx = scheme_source!(terms[$t], cell, cID, cIndex, prev.x, runtime, rho_prev)
                ac, by = scheme_source!(terms[$t], cell, cID, cIndex, prev.y, runtime, rho_prev)
                ac, bz = scheme_source!(terms[$t], cell, cID, cIndex, prev.z, runtime, rho_prev)
                AC += F(ac) # assuming ac's for all directions are equal
                BX += F(bx)
                BY += F(by)
                BZ += F(bz)
            end
            push!(out.args, function_call_scheme_source)
        end
        return quote
            z = zero(F)
            ac = z
            bx = z
            by = z
            bz = z
            AC = z
            BX = z
            BY = z
            BZ = z
            $(out.args...)
            return AC, BX, BY, BZ
        end
    end
end

# Sources generated function definition
@generated function _sources!(
    model::Model{TN,SN,T,S}, sources::SRC, volume::F, cID
    ) where {TN,SN,T,S,SRC,F}
    # Allocate expression array to store source function
    out = Expr(:block)

    # Loop over number of terms and store source function in array
    if S.parameters[1].parameters[1] <: AbstractScalarField
        for s in 1:SN
            expression_call_sources = quote
                (; field, sign) = sources[$s]
                B += F(sign*field[cID]*volume)
            end
            push!(out.args, expression_call_sources)
        end
        return quote
            B = zero(F)
            $(out.args...)
            return B
        end
    elseif S.parameters[1].parameters[1] <: AbstractVectorField
        for s in 1:SN
            expression_call_sources = quote
                (; field, sign) = sources[$s]
                Bx += F(sign*field.x[cID]*volume)
                By += F(sign*field.y[cID]*volume)
                Bz += F(sign*field.z[cID]*volume)
            end
            push!(out.args, expression_call_sources)
        end
        return quote
            z = zero(F)
            Bx = z
            By = z
            Bz = z
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
    ndrange = length(nzval0)
    kernel! = _update_equation!(_setup(backend, workgroup, ndrange)...)
    kernel!(nzval, nzval0)
    # # KernelAbstractions.synchronize(backend)
end

@kernel function _update_equation!(nzval, nzval0) 
    i = @index(Global)

    @inbounds begin
        nzval[i] = nzval0[i]
    end
end
