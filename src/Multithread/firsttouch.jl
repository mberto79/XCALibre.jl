export FirstTouch, first_touch, first_touch_copy, first_touch_zeros, first_touch_enabled

# NEW SECTION: NUMA placement by parallel first touch
# Linux places a page in the NUMA domain of the thread that first writes it. Arrays built on the
# main thread therefore sit in one domain, and on a multi-domain node every thread streams them
# through that domain's memory channels. Copying them with the solver's static partition (chunk c
# written by thread c, as in xmul! and the XVector ops) puts each chunk next to the thread using it.

const FIRST_TOUCH = Ref(false)

"""
    first_touch_enabled()

`true` after `activate_multithread(CPU(static=true); first_touch=true)` on more than one thread:
equations, fields and preconditioners are then allocated by parallel first touch.
"""
first_touch_enabled() = FIRST_TOUCH[] && Threads.nthreads() > 1

# chunk c on thread c; a nested call cannot pin chunks to threads, so it places pages at random
function _touch_chunks(g::G, k) where G
    _in_threaded_region() && @warn "first touch called inside a threaded loop: pages not placed" maxlog=1
    _each_chunk_task(g, k)
end

# start of chunk c's entries: rows reach `a` through start(row), rows beyond n start past its end
function _chunk_starts(start, n, len, k)
    s = [first(_chunk(n, k, c)) > n ? len + 1 : Int(start(first(_chunk(n, k, c)))) for c ∈ 1:k]
    push!(s, len + 1)
    # a reader that leaves empty ranges as 0:0 breaks the offsets; cut by position instead
    s[1] == 1 && issorted(s) ? s : nothing
end

"""
    first_touch_copy(a::Vector)
    first_touch_copy(a::Vector, ranges)

Copy `a` into a new vector whose pages are first written by the threads that work on them. By
default chunk c of `a` is the solver's chunk c. With `ranges` (e.g. `mesh.cell_faces_range`),
`a` is indexed through a range per row, and chunk c holds the entries of the rows in chunk c.
"""
first_touch_copy(a::Vector) = _first_touch_copy(a, identity, length(a))
first_touch_copy(a::Vector, ranges::AbstractVector{<:AbstractRange}) =
    _first_touch_copy(a, i -> first(ranges[i]), length(ranges))

function _first_touch_copy(a::Vector, start, n)
    k, len = Threads.nthreads(), length(a)
    b = similar(a)
    (k == 1 || len < _MIN_THREADED_WORK) && return copyto!(b, a)
    s = something(_chunk_starts(start, n, len, k), _chunk_starts(identity, len, len, k))
    _touch_chunks(k) do c
        @inbounds for i ∈ s[c]:s[c + 1] - 1
            b[i] = a[i]
        end
    end
    b
end

"""
    first_touch_zeros(backend, T, n)

Zero vector of length `n`. With first touch enabled on the CPU, each thread writes the chunk it
later works on; otherwise it is `KernelAbstractions.zeros(backend, T, n)`.
"""
first_touch_zeros(backend, ::Type{T}, n) where T = KernelAbstractions.zeros(backend, T, n)
first_touch_zeros(backend::CPU, ::Type{T}, n) where T = first_touch_enabled() ?
    _first_touch_zeros(T, n) : KernelAbstractions.zeros(backend, T, n)

function _first_touch_zeros(::Type{T}, n) where T
    k, a = Threads.nthreads(), Vector{T}(undef, n)
    n < _MIN_THREADED_WORK && return fill!(a, zero(T))
    _touch_chunks(k) do c
        @inbounds for i ∈ _chunk(n, k, c)
            a[i] = zero(T)
        end
    end
    a
end

"""
    FirstTouch()

Adapt.jl target that copies every `Vector` of a structure by parallel first touch (see
[`first_touch_copy`](@ref)); `first_touch(x)` is `adapt(FirstTouch(), x)`. A mesh cuts its
range-indexed arrays (cell faces, neighbours, ...) by the cell, face or node owning them.
"""
struct FirstTouch end

Adapt.adapt_storage(::FirstTouch, a::Vector) = first_touch_copy(a)

# nzval and colval cut by rowptr, so each thread holds the rows it multiplies in xmul!
Adapt.adapt_structure(to::FirstTouch, A::SparseXCSR{Bi}) where Bi = begin
    B = parent(A)
    start = i -> B.rowptr[i] + 1 - Bi
    SparseXCSR(SparseMatrixCSR{Bi}(B.m, B.n, first_touch_copy(B.rowptr),
        _first_touch_copy(B.colval, start, B.m), _first_touch_copy(B.nzval, start, B.m)))
end

first_touch(x) = Adapt.adapt(FirstTouch(), x)

# construction sites: a copy only when first touch is enabled
_first_touch_if_enabled(x) = first_touch_enabled() ? first_touch(x) : x
