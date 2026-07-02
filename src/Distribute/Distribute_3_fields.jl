export DistributedScalarField, DistributedVectorField
export sync!, pnorm, pdot, pmean

"""
    DistributedScalarField(dmesh, backend; comm=MPI.COMM_WORLD)

A `ScalarField` on a `DistributedMesh` paired with its `HaloExchange`; `sync!` fills ghosts.
"""
struct DistributedScalarField{F<:ScalarField,H<:HaloExchange}
    field::F
    halo::H
end
DistributedScalarField(dmesh::DistributedMesh, backend; comm=MPI.COMM_WORLD) =
    DistributedScalarField(ScalarField(dmesh), HaloExchange(dmesh, 1, backend; comm))

"""
    DistributedVectorField(dmesh, backend; comm=MPI.COMM_WORLD)

A `VectorField` on a `DistributedMesh` paired with a 3-wide `HaloExchange`.
"""
struct DistributedVectorField{F<:VectorField,H<:HaloExchange}
    field::F
    halo::H
end
DistributedVectorField(dmesh::DistributedMesh, backend; comm=MPI.COMM_WORLD) =
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

# ghost centres are verbatim copies, so delegation alone leaves ghosts consistent
initialise!(df::DistributedField, value) = initialise!(df.field, value)

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
