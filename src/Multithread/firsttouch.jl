export FirstTouch, first_touch

# NEW SECTION: NUMA placement by parallel first touch
# Linux places a page in the NUMA domain of the thread that first writes it, so arrays built on the main
# thread sit in one domain; writing each chunk from the thread that uses it (as xmul!) places it locally.

const FIRST_TOUCH = Ref(false)

# true after activate_multithread(CPU(static=true); first_touch=true) on more than one thread:
# equations, fields and preconditioners are then allocated by parallel first touch
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

# copy of `a` whose chunk c is first written by thread c; with `ranges` (e.g. cell_faces_range)
# `a` is indexed through a range per row, and chunk c holds the entries of the rows in chunk c
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

# zero vector; with first touch enabled on the CPU each thread writes the chunk it works on,
# otherwise KernelAbstractions.zeros(backend, T, n)
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

Adapt.jl target that copies every `Vector` of a structure by parallel first touch, so that each
thread writes first the chunk it later works on; see [`first_touch`](@ref).
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

"""
    first_touch(mesh)

Copy `mesh` (or any structure of `Vector`s, e.g. a matrix) so that each thread first writes, and
so places in its NUMA domain, the chunk of cells, faces or rows it later works on. Call it after
pinning the threads and after `activate_multithread(CPU(static=true); first_touch=true)`, and
build the model from the returned mesh: `mesh = first_touch(mesh)`. Pays on multi-socket (NUMA)
nodes; results are unchanged.
"""
first_touch(x) = Adapt.adapt(FirstTouch(), x)

# construction sites: a copy only when first touch is enabled
_first_touch_if_enabled(x) = first_touch_enabled() ? first_touch(x) : x
