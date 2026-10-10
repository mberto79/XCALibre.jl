export FirstTouch, first_touch

# NEW SECTION: NUMA placement by parallel first touch
# Linux places a page in the NUMA domain of the thread that first writes it; KernelAbstractions'
# zeros/copyto! on CPU(static=true) write chunk c from thread c, as the solver's loops read it.

# backend that CPU arrays are built with; set by activate_multithread
const CPU_BACKEND = Ref(CPU())
_active_backend(backend) = backend
_active_backend(::CPU) = CPU_BACKEND[]

"""
    FirstTouch()

Adapt.jl target that copies every `Array` of a structure with the backend given to
[`activate_multithread`](@ref); see [`first_touch`](@ref).
"""
struct FirstTouch end

Adapt.adapt_storage(::FirstTouch, a::Array) = begin
    backend = CPU_BACKEND[]
    KernelAbstractions.copyto!(backend, KernelAbstractions.allocate(backend, eltype(a), size(a)), a)
end

Adapt.adapt_structure(to::FirstTouch, A::SparseXCSR{Bi}) where Bi = begin
    B = parent(A)
    SparseXCSR(SparseMatrixCSR{Bi}(B.m, B.n, Adapt.adapt(to, B.rowptr), Adapt.adapt(to, B.colval),
        Adapt.adapt(to, B.nzval)))
end

"""
    first_touch(mesh)

Copy `mesh` (or any structure of `Array`s) in parallel so that, with `CPU(static=true)` passed to
[`activate_multithread`](@ref) and pinned threads, each thread first writes, and so places in its
NUMA domain, the chunk of cells, faces or nodes it later works on. The mesh is read on one thread,
so call it before building the model: `mesh = first_touch(mesh)`. Pays on multi-socket (NUMA)
nodes; results are unchanged.
"""
first_touch(x) = Adapt.adapt(FirstTouch(), x)
