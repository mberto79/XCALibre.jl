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

# below this many elements touched, one fork/join costs more than the loop saves
const _MIN_THREADED_WORK = 1 << 16

# nested regions cannot pin chunks to threads, so they run dynamically
_in_threaded_region() = ccall(:jl_in_threaded_region, Cint, ()) != 0

# chunk c on thread c as Base's :static, but spawned as a tree: a task hands the upper half of its chunks
# to the thread of that half's first chunk, so no thread spawns more than log2(k) tasks
function _each_chunk_task(g::G, k) where G
    if !_in_threaded_region() && k == Threads.threadpoolsize()
        off = Threads.threadpoolsize(:interactive)
        ccall(:jl_enter_threaded_region, Cvoid, ())
        try
            Threads.threadid() == off + 1 ? _run_chunks(g, 1, k, off) : wait(_sticky(() -> _run_chunks(g, 1, k, off), off + 1))
        finally
            ccall(:jl_exit_threaded_region, Cvoid, ())
        end
        return nothing
    end
    Threads.@threads :dynamic for c ∈ 1:k
        g(c)
    end
    nothing
end

function _run_chunks(g::G, lo, hi, off) where G
    tasks = Task[]
    while hi > lo
        mid = (lo + hi + 1) >> 1
        push!(tasks, let mid = mid, hi = hi
            _sticky(() -> _run_chunks(g, mid, hi, off), off + mid)
        end)
        hi = mid - 1
    end
    try
        g(lo)
    finally
        foreach(wait, tasks)
    end
end

_sticky(f, tid) = begin
    t = Task(f)
    t.sticky = true
    ccall(:jl_set_task_tid, Cint, (Any, Cint), t, tid - 1)
    schedule(t)
end

@inline function _foreach_chunk(f, n, work=n)
    k = Threads.nthreads()
    (k == 1 || work < _MIN_THREADED_WORK) && return f(1:n)
    _each_chunk_task(c -> f(_chunk(n, k, c)), k)
end

# partials one cache line apart, summed in chunk order: deterministic for a fixed thread count
function _reduce_chunks(f, n, ::Type{T}) where T
    k = Threads.nthreads()
    (k == 1 || n < _MIN_THREADED_WORK) && return f(1:n)
    stride = max(1, 64 ÷ sizeof(T))
    partials = Vector{T}(undef, stride*k)
    _each_chunk_task(k) do c
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

# half-precision sums stagnate after a few hundred terms, so they accumulate in Float32
_acc_type(::Type{T}) where T = T
_acc_type(::Type{Float16}) = Float32

function Krylov.kdot(n::Integer, x::XVector{T}, y::XVector{T}) where T<:AbstractFloat
    xd, yd = x.data, y.data
    acc = _reduce_chunks(n, _acc_type(T)) do r
        # derived from xd, not captured: a captured type is a DataType field on Julia 1.10
        TA = _acc_type(eltype(xd))
        s = zero(TA)
        @inbounds @simd for i ∈ r
            s += TA(xd[i])*TA(yd[i])
        end
        s
    end
    T(acc)
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
