# NEW SECTION: threaded dense vector for the Krylov solvers
struct XVector{T} <: DenseVector{T}
    data::Vector{T}
end

Base.parent(x::XVector) = x.data
Base.size(x::XVector) = size(x.data)
Base.IndexStyle(::Type{<:XVector}) = IndexLinear()
@inline Base.getindex(x::XVector, i::Int) = (@boundscheck checkbounds(x.data, i); @inbounds x.data[i])
@inline Base.setindex!(x::XVector, v, i::Int) = (@boundscheck checkbounds(x.data, i); @inbounds x.data[i] = v)
Base.similar(x::XVector, ::Type{T}, dims::Tuple{Int}) where T = XVector(similar(x.data, T, dims))
Base.unsafe_convert(::Type{Ptr{T}}, x::XVector{T}) where T = Base.unsafe_convert(Ptr{T}, x.data)
Base.elsize(::Type{XVector{T}}) where T = sizeof(T)
Base.strides(::XVector) = (1,)
KernelAbstractions.get_backend(::XVector) = CPU()

# chunked as xmul! chunks rows; chunk c always runs on thread c so it reads back what it wrote
@inline function _chunk(n, k, c)
    d, r = divrem(n, k)
    (c - 1)*d + min(c - 1, r) + 1 : c*d + min(c, r)
end

@inline function _foreach_chunk(f, n)
    k = Threads.nthreads()
    k == 1 && return f(1:n)
    Threads.@threads :static for c ∈ 1:k
        f(_chunk(n, k, c))
    end
    nothing
end

# partials one cache line apart, summed in chunk order: deterministic for a fixed thread count
function _reduce_chunks(f, n, ::Type{T}) where T
    k = Threads.nthreads()
    k == 1 && return f(1:n)
    stride = max(1, 64 ÷ sizeof(T))
    partials = Vector{T}(undef, stride*k)
    Threads.@threads :static for c ∈ 1:k
        @inbounds partials[stride*(c - 1) + 1] = f(_chunk(n, k, c))
    end
    s = zero(T)
    @inbounds for c ∈ 1:k
        s += partials[stride*(c - 1) + 1]
    end
    s
end

function Base.fill!(x::XVector{T}, v) where T
    d, val = x.data, convert(T, v)
    _foreach_chunk(length(d)) do r
        @inbounds for i ∈ r
            d[i] = val
        end
    end
    x
end

function Base.copyto!(y::XVector, x::XVector)
    length(y) == length(x) || throw(DimensionMismatch())
    yd, xd = y.data, x.data
    _foreach_chunk(length(yd)) do r
        @inbounds for i ∈ r
            yd[i] = xd[i]
        end
    end
    y
end

# NEW SECTION: Krylov.jl vector primitives

function Krylov.kdot(n::Integer, x::XVector{T}, y::XVector{T}) where T<:AbstractFloat
    xd, yd = x.data, y.data
    _reduce_chunks(n, T) do r
        s = zero(T)
        @inbounds @simd for i ∈ r
            s += xd[i]*yd[i]
        end
        s
    end
end

Krylov.knorm(n::Integer, x::XVector{T}) where T<:AbstractFloat = sqrt(Krylov.kdot(n, x, x))

function Krylov.kscal!(n::Integer, s::T, x::XVector{T}) where T<:AbstractFloat
    xd = x.data
    _foreach_chunk(n) do r
        @inbounds @simd for i ∈ r
            xd[i] *= s
        end
    end
    x
end

Krylov.kcopy!(n::Integer, y::XVector{T}, x::XVector{T}) where T<:AbstractFloat = copyto!(y, x)

function Krylov.kscalcopy!(n::Integer, y::XVector{T}, s::T, x::XVector{T}) where T<:AbstractFloat
    yd, xd = y.data, x.data
    _foreach_chunk(n) do r
        @inbounds @simd for i ∈ r
            yd[i] = s*xd[i]
        end
    end
    y
end

function Krylov.kdivcopy!(n::Integer, y::XVector{T}, x::XVector{T}, s::T) where T<:AbstractFloat
    yd, xd = y.data, x.data
    _foreach_chunk(n) do r
        @inbounds @simd for i ∈ r
            yd[i] = xd[i]/s
        end
    end
    y
end

function Krylov.kaxpy!(n::Integer, s::T, x::XVector{T}, y::XVector{T}) where T<:AbstractFloat
    yd, xd = y.data, x.data
    _foreach_chunk(n) do r
        @inbounds @simd for i ∈ r
            yd[i] += s*xd[i]
        end
    end
    y
end

function Krylov.kaxpby!(n::Integer, s::T, x::XVector{T}, t::T, y::XVector{T}) where T<:AbstractFloat
    yd, xd = y.data, x.data
    _foreach_chunk(n) do r
        @inbounds @simd for i ∈ r
            yd[i] = s*xd[i] + t*yd[i]
        end
    end
    y
end

Krylov.kfill!(x::XVector{T}, val::T) where T<:AbstractFloat = fill!(x, val)
