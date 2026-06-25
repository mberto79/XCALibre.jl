export setField_Box!, setField_Circle2D!, setField_Sphere3D!, setField_Expression!

"""
    setField_Box!(; mesh, field, value::F, min_corner::V, max_corner::V)

Sets field values to `value` for all cells whose centre lies within the axis-aligned
box defined by `min_corner` and `max_corner`.  Runs on the same backend as `field`.

Returns the number of cells set.
"""
function setField_Box!(; mesh, field, value::F, min_corner::V, max_corner::V) where {F <: AbstractFloat, V <: AbstractVector}
    @assert length(min_corner) == 3 "`min_corner` must have exactly 3 elements"
    @assert length(max_corner) == 3 "`max_corner` must have exactly 3 elements"

    backend = KA.get_backend(field)
    ndrange  = length(mesh.cells)
    lo    = SVector{3,F}(min_corner[1], min_corner[2], min_corner[3])
    hi    = SVector{3,F}(max_corner[1], max_corner[2], max_corner[3])
    count = KA.zeros(backend, Int64, 1)
    kernel! = _setField_Box!(_setup(backend, 64, ndrange)...)
    kernel!(field, mesh.cells, F(value), lo, hi, count)
    KA.synchronize(backend)
    return Int(sum(count))
end

@kernel function _setField_Box!(field, cells, value, lo, hi, count)
    i = @index(Global)
    @inbounds begin
        c = cells[i].centre
        if lo[1] <= c[1] <= hi[1] && lo[2] <= c[2] <= hi[2] && lo[3] <= c[3] <= hi[3]
            field[i] = value
            Atomix.@atomic count[] += one(Int64)
        end
    end
end

"""
    setField_Circle2D!(; mesh, field, value::F, centre::V, radius::F)

Sets field values to `value` for cells whose centre is within `radius` of `centre`
in the X-Y plane (z coordinate ignored).  Runs on the same backend as `field`.

Returns the number of cells set.
"""
function setField_Circle2D!(; mesh, field, value::F, centre::V, radius::F) where {F <: AbstractFloat, V <: AbstractVector}
    @assert length(centre) == 2 "`centre` must have exactly 2 elements. Use `setField_Sphere3D!` for 3-D."

    backend = KA.get_backend(field)
    ndrange  = length(mesh.cells)
    c0    = SVector{3,F}(centre[1], centre[2], zero(F))
    count = KA.zeros(backend, Int64, 1)
    kernel! = _setField_Sphere!(_setup(backend, 64, ndrange)...)
    kernel!(field, mesh.cells, F(value), c0, F(radius), count)
    KA.synchronize(backend)
    return Int(sum(count))
end

"""
    setField_Sphere3D!(; mesh, field, value::F, centre::V, radius::F)

Sets field values to `value` for cells whose centre lies within `radius` of `centre`.
Runs on the same backend as `field`.

Returns the number of cells set.
"""
function setField_Sphere3D!(; mesh, field, value::F, centre::V, radius::F) where {F <: AbstractFloat, V <: AbstractVector}
    @assert length(centre) == 3 "`centre` must have exactly 3 elements. Use `setField_Circle2D!` for 2-D."

    backend = KA.get_backend(field)
    ndrange  = length(mesh.cells)
    c0    = SVector{3,F}(centre[1], centre[2], centre[3])
    count = KA.zeros(backend, Int64, 1)
    kernel! = _setField_Sphere!(_setup(backend, 64, ndrange)...)
    kernel!(field, mesh.cells, F(value), c0, F(radius), count)
    KA.synchronize(backend)
    return Int(sum(count))
end

@kernel function _setField_Sphere!(field, cells, value, centre, radius, count)
    i = @index(Global)
    @inbounds begin
        c = cells[i].centre
        if norm(c - centre) <= radius
            field[i] = value
            Atomix.@atomic count[] += one(Int64)
        end
    end
end

"""
    setField_Expression!(; mesh, field, condition, value_true::F, value_false=nothing)

Assigns `value_true` to every cell whose centre `(x, y, z)` satisfies `condition(x, y, z)`.
If `value_false` is provided all other cells receive `value_false`; otherwise they are unchanged.
Runs on the same backend as `field` (CPU or GPU).
For GPU backends, `condition` must be callable from a device kernel.

Returns the number of cells where `condition` returned `true`.

# Example — Rayleigh-Taylor interface
```julia
Ly = 4.0; A = 0.05; λ = 1.0
setField_Expression!(
    mesh      = mesh,
    field     = model.fluid.alpha,
    condition = (x, y, z) -> y > Ly/2 + A * cos(2π * x / λ),
    value_true  = 1.0,
    value_false = 0.0,
)
```
"""
function setField_Expression!(;
    mesh,
    field,
    condition::Cond,
    value_true::F,
    value_false::Union{F,Nothing} = nothing,
) where {F <: AbstractFloat, Cond <: Function}
    backend  = KA.get_backend(field)
    ndrange  = length(mesh.cells)
    vf       = value_false !== nothing ? F(value_false) : zero(F)
    has_vf   = value_false !== nothing
    count    = KA.zeros(backend, Int64, 1)
    kernel!  = _setField_Expression!(_setup(backend, 64, ndrange)...)
    kernel!(field, mesh.cells, condition, F(value_true), vf, has_vf, count)
    KA.synchronize(backend)
    return Int(sum(count))
end

@kernel function _setField_Expression!(field, cells, condition::Cond, value_true, value_false, has_vf, count) where Cond
    i = @index(Global)
    @inbounds begin
        c = cells[i].centre
        if condition(c[1], c[2], c[3])
            field[i] = value_true
            Atomix.@atomic count[] += one(Int64)
        elseif has_vf
            field[i] = value_false
        end
    end
end

"""
    setField_Expression!(; mesh, field, condition::Function, value_true::F,
                          value_false::Union{F,Nothing}=nothing
                        ) where {F <: AbstractFloat}

Assigns `value_true` to every cell whose centre `(x, y, z)` satisfies the user-supplied condition.
If `value_false` is provided, all other cells get `value_false`; otherwise they are left unchanged.

The `condition` can capture any variables from the calling scope via Julia's closure
syntax - useful for parametric initial conditions like a sinusoidal interface for
Rayleigh-Taylor or any expression that doesn't fit a box/sphere/circle.

# Input arguments

- `mesh` reference to `domain` inside a `Physics` model defined by the user.
- `field` field to be modified.
- `condition` function `(x, y, z) -> Bool` evaluated at each cell centre.
- `value_true` value assigned to cells where the predicate returns `true`.
- `value_false` (optional) value assigned where the predicate returns `false`. If
  omitted, those cells are left unchanged.

# Example — Rayleigh-Taylor interface

```julia
Ly = 4.0; A = 0.05; λ = 1.0
setField_Expression!(
    mesh      = mesh,
    field     = model.fluid.alpha,
    condition = (x, y, z) -> y > Ly/2 + A * cos(2π * x / λ),
    value_true  = 1.0,
    value_false = 0.0,
)
```
"""
function setField_Expression!(;
    mesh,
    field,
    condition::Function,
    value_true::F,
    value_false::Union{F,Nothing} = nothing,
) where {F <: AbstractFloat}

    cells_in_region = Int[]
    cells_outside   = Int[]

    for (id, cell) in enumerate(mesh.cells)
        c = cell.centre
        if condition(c[1], c[2], c[3])
            push!(cells_in_region, id)
        else
            push!(cells_outside, id)
        end
    end

    if !isempty(cells_in_region)
        field.values[cells_in_region] .= value_true
    end
    if value_false !== nothing && !isempty(cells_outside)
        field.values[cells_outside] .= value_false
    end

    return length(cells_in_region)
end