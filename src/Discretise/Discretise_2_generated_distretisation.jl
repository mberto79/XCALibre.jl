export discretise!, update_equation!

# NEW SECTION: kernel arguments
# Kernel arguments are copied by value into per-thread local memory, so terms, sources and fields
# reach the discretise kernels without their mesh; phi keeps only the face_gDiff column schemes read.
_kernel_field(f, m=()) = f
_kernel_field(f::ScalarField, m=()) = ScalarField(f.values, m)
_kernel_field(f::FaceScalarField, m=()) = FaceScalarField(f.values, m)
_kernel_field(f::VectorField, m=()) = VectorField(_kernel_field(f.x), _kernel_field(f.y), _kernel_field(f.z), m)
_kernel_field(f::FaceVectorField, m=()) =
    FaceVectorField(_kernel_field(f.x), _kernel_field(f.y), _kernel_field(f.z), m)
_kernel_field(t::TensorField, m=()) = TensorField(
    _kernel_field(t.xx), _kernel_field(t.xy), _kernel_field(t.xz),
    _kernel_field(t.yx), _kernel_field(t.yy), _kernel_field(t.yz),
    _kernel_field(t.zx), _kernel_field(t.zy), _kernel_field(t.zz), m)

_kernel_model(model, mesh) = begin
    m = (; face_gDiff=mesh.face_gDiff)
    terms = map(t -> Operator(
        _kernel_field(t.flux), _kernel_field(t.phi, m), t.sign, _kernel_type(t.type),
        _kernel_field(t.grad)), model.terms)
    sources = map(s -> Src(_kernel_field(s.field), s.sign), model.sources)
    terms, sources
end

function discretise!(
    eqn::ModelEquation{T,M,E,S,P}, prev, config; rho_prev=eqn.model.terms[1].flux, time=nothing
    ) where {T<:VectorModel,M,E,S,P}
    (; hardware, runtime) = config
    (; backend, workgroup) = hardware
    prev = time_level!(eqn.model.terms[1], prev, rho_prev, time, config)

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
    (; diag_nz, face_nz) = eqn.equation

    _pattern_extended(nzval0, mesh) && fill_nzval!(nzval0, config)

    _sync_gradients!(model.terms, mesh, config)
    terms, sources = _kernel_model(model, mesh)
    (; cells, faces, cell_faces, cell_neighbours, cell_nsign) = mesh
    ndrange = length(cells)
    kernel! = _sized(_discretise_vector_model!, backend, workgroup, ndrange)
    kernel!(terms, sources, cells, faces, cell_faces, cell_neighbours, cell_nsign, nzval0,
        diag_nz, face_nz, bx, by, bz, _kernel_field(prev), runtime, _kernel_field(rho_prev))
    # # KernelAbstractions.synchronize(backend)
end

@kernel function _discretise_vector_model!(
    terms::TERMS, sources::SRCS, cells, faces, cell_faces, cell_neighbours, cell_nsign,
    nzval0::AbstractArray{F}, diag_nz, face_nz, bx, by, bz, prev, runtime, rho_prev) where {F,TERMS,SRCS}
    i = @index(Global)

    @inbounds begin
        # Define workitem cell and extract required fields
        faces_range = cells.faces_range[i]
        volume = cells.volume[i]


        cIndex = diag_nz[i]

        # For loop over workitem cell faces
        ac_sum = zero(F)
        bf = _face_source_init(terms, zero(SVector{3,F}))
        for fi in faces_range
            # Retrieve indices for discretisation
            fID = cell_faces[fi]
            ns = cell_nsign[fi] # normal sign
            nID = cell_neighbours[fi]
            nIndex = face_nz[fi]


            # Call scheme generated fucntion
            ac, an = _scheme!(terms, nzval0, cells, faces, nID, ns, cIndex, nIndex, fID, prev, runtime)
            ac_sum += ac
            nzval0[nIndex] = an
            bf = _scheme_face_source!(terms, bf, cells, faces, i, nID, ns, fID)
        end

        
        # Call scheme source generated function NEEDS UPDATING!
        ac, bx1, by1, bz1 = _scheme_source!(terms, cells, i, cIndex, prev, runtime, rho_prev)
        
        nzval0[cIndex] = ac_sum + ac

        # Call sources generated function
        bx2, by2, bz2 = _sources!(sources, volume, i)
        bx[i] = _add_face_source(bx1 + bx2, bf, 1)
        by[i] = _add_face_source(by1 + by2, bf, 2)
        bz[i] = _add_face_source(bz1 + bz2, bf, 3)
    end
end

function discretise!(
    eqn::ModelEquation{T,M,E,S,P}, prev, config; rho_prev=eqn.model.terms[1].flux, time=nothing
    ) where {T<:ScalarModel,M,E,S,P}

    (; hardware, runtime) = config
    (; backend, workgroup) = hardware
    prev = time_level!(eqn.model.terms[1], prev, rho_prev, time, config)

    # Retrieve variabels for defition
    mesh = eqn.model.terms[1].phi.mesh
    model = eqn.model

    # Sparse array and b accessor call
    A = _A(eqn)
    b = _b(eqn)

    # Sparse array fields accessors
    nzval = _nzval(A)
    (; diag_nz, face_nz) = eqn.equation

    _pattern_extended(nzval, mesh) && fill_nzval!(nzval, config)

    _sync_gradients!(model.terms, mesh, config)
    terms, sources = _kernel_model(model, mesh)
    (; cells, faces, cell_faces, cell_neighbours, cell_nsign) = mesh
    ndrange = length(cells)
    kernel! = _sized(_discretise_scalar_model!, backend, workgroup, ndrange)
    kernel!(terms, sources, cells, faces, cell_faces, cell_neighbours, cell_nsign, nzval,
        diag_nz, face_nz, b, _kernel_field(prev), runtime, _kernel_field(rho_prev))
    # # KernelAbstractions.synchronize(backend)
end

# the kernels assign every diagonal and one entry per cell face, which is the whole pattern unless a
# boundary condition added entries (periodic) or two faces share a cell pair; only then is a reset needed
_pattern_extended(nzval, mesh) = length(nzval) != length(mesh.cells) + length(mesh.cell_faces)

fill_nzval!(nzval, config) = begin
    z = zero(eltype(nzval))
    xcal_foreach(nzval, config) do i
        nzval[i] = z
    end
end

@kernel function _discretise_scalar_model!(
    terms::TERMS, sources::SRCS, cells, faces, cell_faces, cell_neighbours, cell_nsign,
    nzval::AbstractArray{F}, diag_nz, face_nz, b, prev, runtime, rho_prev) where {F,TERMS,SRCS}

    i = @index(Global)

    @inbounds begin
        # Define workitem cell and extract required fields
        faces_range = cells.faces_range[i]
        volume = cells.volume[i]

        cIndex = diag_nz[i]

        # For loop over workitem cell faces
        ac_sum = zero(F)
        bf = _face_source_init(terms, zero(F))
        for fi in faces_range
            # Retrieve indices for discretisation
            fID = cell_faces[fi]
            ns = cell_nsign[fi] # normal sign
            nID = cell_neighbours[fi]
            nIndex = face_nz[fi]

            # Call scheme generated fucntion
            ac, an = _scheme!(terms, nzval, cells, faces, nID, ns, cIndex, nIndex, fID, prev, runtime)
            ac_sum += ac
            nzval[nIndex] = an
            bf = _scheme_face_source!(terms, bf, cells, faces, i, nID, ns, fID)
        end
        
        # Call scheme source generated function
        ac, b1 = _scheme_source!(terms, cells, i, cIndex, prev, runtime, rho_prev)
        nzval[cIndex] = ac_sum + ac

        # Call sources generated function
        b2 = _sources!(sources, volume, i)
        b[i] = _add_face_source(b2 + b1, bf, 1)
    end
end

return_quote(x, t) = :(nothing)

# Scheme generated function definition
@generated function _scheme!(
    terms::TERMS, nzval::AbstractArray{F}, cells, faces,
    nID, ns, cIndex, nIndex, fID, prev, runtime
    ) where {TERMS,F}
    TN = fieldcount(TERMS)
    # Allocate expression array to store scheme function
    out = Expr(:block)

    # Loop over number of terms and store scheme function in array
    for t in 1:TN
        function_call_scheme = quote
            ac, an = scheme!(terms[$t], nzval, cells, faces, nID, ns, cIndex, nIndex, fID, prev, runtime)
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

# Face source: `bf` accumulates the explicit face contributions of the operators with `has_face_source`
# (SVector{3} for vector, scalar otherwise); it is `nothing` when no operator has one, so equations
# without second-order schemes carry no extra work.
@generated function _face_source_init(terms::TERMS, z) where {TERMS}
    any(t -> has_face_source(fieldtype(TERMS, t)), 1:fieldcount(TERMS)) ? :(z) : :(nothing)
end

@generated function _scheme_face_source!(
    terms::TERMS, bf, cells, faces, cID, nID, ns, fID) where {TERMS}
    idx = [t for t in 1:fieldcount(TERMS) if has_face_source(fieldtype(TERMS, t))]
    isempty(idx) && return :(bf)
    calls = [:(bf += scheme_face_source!(terms[$t], cells, faces, cID, nID, ns, fID)) for t in idx]
    quote
        $(calls...)
        return bf
    end
end

@inline _add_face_source(b, ::Nothing, k) = b
@inline _add_face_source(b, bf::AbstractVector, k) = b + bf[k]
@inline _add_face_source(b, bf::Number, k) = b + bf

# Ghost gradients are not consistent on other ranks; distributed meshes extend this hook
sync_gradient!(grad, mesh, config) = nothing

@generated function _sync_gradients!(terms::TERMS, mesh, config) where {TERMS}
    idx = [t for t in 1:fieldcount(TERMS) if has_face_source(fieldtype(TERMS, t))]
    isempty(idx) && return :(nothing)
    calls = [:(sync_gradient!(terms[$t].grad, mesh, config)) for t in idx]
    :($(calls...); nothing)
end

# Scheme source generated function definition
@generated function _scheme_source!(terms::TERMS, cells::AbstractVector{<:Cell{F}}, cID, cIndex, prev::P, runtime, rho_prev) where {TERMS,F,P}
    TN = fieldcount(TERMS)
    # Allocate expression array to store scheme_source function
    out = Expr(:block)
    
    # Loop over number of terms and store scheme_source function in array
    if !(P <: AbstractVectorField)
        for t in 1:TN
            function_call_scheme_source = quote
                ac, b = scheme_source!(terms[$t], cells, cID, cIndex, prev, runtime, rho_prev)
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
    else
        for t in 1:TN
            function_call_scheme_source = quote
                ac, bx = scheme_source!(terms[$t], cells, cID, cIndex, prev.x, runtime, rho_prev)
                ac, by = scheme_source!(terms[$t], cells, cID, cIndex, prev.y, runtime, rho_prev)
                ac, bz = scheme_source!(terms[$t], cells, cID, cIndex, prev.z, runtime, rho_prev)
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
    sources::SRC, volume::F, cID
    ) where {SRC,F}
    SN = fieldcount(SRC)
    # Allocate expression array to store source function
    out = Expr(:block)

    # Loop over number of terms and store source function in array
    if SRC.parameters[1].parameters[1] <: AbstractScalarField
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
    elseif SRC.parameters[1].parameters[1] <: AbstractVectorField
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
    kernel! = _sized(_update_equation!, backend, workgroup, ndrange)
    kernel!(nzval, nzval0)
    # # KernelAbstractions.synchronize(backend)
end

@kernel function _update_equation!(nzval, nzval0) 
    i = @index(Global)

    @inbounds begin
        nzval[i] = nzval0[i]
    end
end
