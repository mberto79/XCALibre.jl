export DistributedScalarField, DistributedVectorField
export sync!, pnorm, pdot, pmean

import XCALibre.Solve: sync!

"""
    DistributedScalarField(dmesh, backend; comm=dmesh.comm)

A `ScalarField` on a `DistributedMesh` paired with its `HaloExchange`; `sync!` fills ghosts.
"""
struct DistributedScalarField{F<:ScalarField,H<:HaloExchange}
    field::F
    halo::H
end
DistributedScalarField(dmesh::DistributedMesh, backend; comm=getfield(dmesh, :comm)) =
    DistributedScalarField(ScalarField(dmesh), HaloExchange(dmesh, 1, backend; comm))

"""
    DistributedVectorField(dmesh, backend; comm=dmesh.comm)

A `VectorField` on a `DistributedMesh` paired with a 3-wide `HaloExchange`.
"""
struct DistributedVectorField{F<:VectorField,H<:HaloExchange}
    field::F
    halo::H
end
DistributedVectorField(dmesh::DistributedMesh, backend; comm=getfield(dmesh, :comm)) =
    DistributedVectorField(VectorField(dmesh), HaloExchange(dmesh, 3, backend; comm))

const DistributedField = Union{DistributedScalarField,DistributedVectorField}

"""
    sync!(df, config)

Halo-exchange the wrapped field's ghost entries.
"""
sync!(df::DistributedField, config) = begin
    (; backend, workgroup) = config.hardware
    halo_exchange!(df.field, df.halo, backend, workgroup)
    nothing
end

# one halo schedule per mesh and width (scalar 1, vector 3, scalar+vector 4), built on first call and shared by
# every field and equation; halo_exchange! is the function barrier past the untyped cache slot
@inline function sync!(x::AbstractScalarField, dm::DistributedMesh, config)
    (; backend, workgroup) = config.hardware
    hc = getfield(dm, :halos)
    hc.w1 === nothing && (hc.w1 = HaloExchange(dm, 1, backend))
    halo_exchange!(x, hc.w1, backend, workgroup)
    nothing
end
@inline function sync!(x::AbstractVectorField, dm::DistributedMesh, config)
    (; backend, workgroup) = config.hardware
    hc = getfield(dm, :halos)
    hc.w3 === nothing && (hc.w3 = HaloExchange(dm, 3, backend))
    halo_exchange!(x, hc.w3, backend, workgroup)
    nothing
end
@inline function sync!(x::Tuple{AbstractScalarField,AbstractVectorField}, dm::DistributedMesh, config)
    (; backend, workgroup) = config.hardware
    hc = getfield(dm, :halos)
    hc.w4 === nothing && (hc.w4 = HaloExchange(dm, 4, backend))
    halo_exchange!(x, hc.w4, backend, workgroup)
    nothing
end
# tensor (and other) ghosts are never consumed off-rank — mirrors grad!'s tensor path
@inline sync!(x, dm::DistributedMesh, config) = nothing

# ghost centres are verbatim copies, so delegation alone leaves ghosts consistent
initialise!(df::DistributedField, value) = initialise!(df.field, value)

# NEW SECTION: ghost consistency check (debug aid, test-only cost)

export check_ghosts

"""
    check_ghosts(x, dm::DistributedMesh, config)

Largest absolute difference, over every rank, between the ghost entries of the scalar or vector
field `x` and the values their owning ranks hold. Zero means every ghost is in sync; anything else
names a primitive that changed a field without `sync!`. One exchange into a scratch copy per call.
"""
function check_ghosts(x::AbstractScalarField, dm::DistributedMesh, config)
    y = ScalarField(dm)
    copyto!(y.values, x.values)
    sync!(y, dm, config)
    _ghost_mismatch(y.values, x.values, dm)
end
function check_ghosts(x::AbstractVectorField, dm::DistributedMesh, config)
    y = VectorField(dm)
    copyto!(y.x.values, x.x.values); copyto!(y.y.values, x.y.values); copyto!(y.z.values, x.z.values)
    sync!(y, dm, config)
    max(_ghost_mismatch(y.x.values, x.x.values, dm), _ghost_mismatch(y.y.values, x.y.values, dm),
        _ghost_mismatch(y.z.values, x.z.values, dm))
end
function _ghost_mismatch(a, b, dm)
    p = getfield(dm, :partition)
    g = p.n_owned+1:p.n_owned+p.n_ghost
    d = maximum(abs.(Array(view(a, g)) .- Array(view(b, g))); init=zero(eltype(a)))
    MPI.Allreduce(d, max, getfield(dm, :comm))
end

_partition(df::DistributedField) = df.field.mesh.partition

# NEW SECTION: global reductions (owned entries only; ghosts never enter reductions)

"""
    pnorm(df::DistributedScalarField)

Global 2-norm over owned entries (`MPI.Allreduce`).
"""
pnorm(df::DistributedScalarField) = begin
    n = _partition(df).n_owned
    s = sum(abs2, view(df.field.values, 1:n); init=zero(eltype(df.field)))
    sqrt(MPI.Allreduce(s, +, df.halo.comm))
end

"""
    pdot(a::DistributedScalarField, b::DistributedScalarField)

Global dot product over owned entries (`MPI.Allreduce`).
"""
pdot(a::DistributedScalarField, b::DistributedScalarField) = begin
    n = _partition(a).n_owned
    s = dot(view(a.field.values, 1:n), view(b.field.values, 1:n))
    MPI.Allreduce(s, +, a.halo.comm)
end

"""
    pmean(df::DistributedScalarField)

Global mean over owned entries (`MPI.Allreduce`).
"""
pmean(df::DistributedScalarField) = begin
    p = _partition(df)
    s = sum(view(df.field.values, 1:p.n_owned); init=zero(eltype(df.field)))
    MPI.Allreduce(s, +, df.halo.comm) / MPI.Allreduce(p.n_owned, +, df.halo.comm)
end
