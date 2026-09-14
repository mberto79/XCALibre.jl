export setField_Box!, setField_Circle2D!, setField_Sphere3D!, setField_Expression!

"""
    setField_Box!(; mesh, field, value::F, min_corner::V, max_corner::V, hardware)

Sets field values to `value` for all cells whose centre lies within the axis-aligned
box defined by `min_corner` and `max_corner`.  Runs on `hardware.backend`, using
`hardware.workgroup` for kernel launch sizing (same convention as the rest of the
package — pass the `hardware` used to build the model/config).

Warning: if a cell's outer boundary extends outside the box but its centre lies
within it, that cell is still counted.

Returns the number of cells set.
"""
function setField_Box!(; mesh, field, value::F, min_corner::V, max_corner::V, hardware) where {F <: AbstractFloat, V <: AbstractVector}
    @assert length(min_corner) == 3 "`min_corner` must have exactly 3 elements"
    @assert length(max_corner) == 3 "`max_corner` must have exactly 3 elements"
    @assert length(mesh.cells) == length(field.mesh.cells) "`mesh` and `field` must be defined on the same domain"

    (; backend, workgroup) = hardware
    cells   = field.mesh.cells
    ndrange = length(cells)
    lo      = SVector{3,F}(min_corner[1], min_corner[2], min_corner[3])
    hi      = SVector{3,F}(max_corner[1], max_corner[2], max_corner[3])
    matched = KA.zeros(backend, Int64, ndrange)
    kernel! = _setField_Box!(_setup(backend, workgroup, ndrange)...)
    kernel!(field, cells, F(value), lo, hi, matched)
    KA.synchronize(backend)
    return Int(sum(matched))
end

@kernel function _setField_Box!(field, cells, value, lo, hi, matched)
    i = @index(Global)
    @inbounds begin
        c = cells[i].centre
        if lo[1] <= c[1] <= hi[1] && lo[2] <= c[2] <= hi[2] && lo[3] <= c[3] <= hi[3]
            field[i] = value
            matched[i] = one(Int64)
        end
    end
end

"""
    setField_Circle2D!(; mesh, field, value::F, centre::V, radius::F, hardware)

Sets field values to `value` for cells whose centre is within `radius` of `centre`
(given as `[x, y]`, assumed to lie in the X-Y plane at z=0). The comparison is made
against each cell's full 3-D centre, so a cell whose centre has a non-zero z-offset
is measured with that offset intact — this is only exact for meshes lying in the
X-Y plane. Runs on `hardware.backend`, using `hardware.workgroup` for kernel launch
sizing.

Returns the number of cells set.
"""
function setField_Circle2D!(; mesh, field, value::F, centre::V, radius::F, hardware) where {F <: AbstractFloat, V <: AbstractVector}
    @assert length(centre) == 2 "`centre` must have exactly 2 elements. Use `setField_Sphere3D!` for 3-D."
    @assert length(mesh.cells) == length(field.mesh.cells) "`mesh` and `field` must be defined on the same domain"

    (; backend, workgroup) = hardware
    cells   = field.mesh.cells
    ndrange = length(cells)
    c0      = SVector{3,F}(centre[1], centre[2], zero(F))
    matched = KA.zeros(backend, Int64, ndrange)
    kernel! = _setField_Sphere!(_setup(backend, workgroup, ndrange)...)
    kernel!(field, cells, F(value), c0, F(radius), matched)
    KA.synchronize(backend)
    return Int(sum(matched))
end

"""
    setField_Sphere3D!(; mesh, field, value::F, centre::V, radius::F, hardware)

Sets field values to `value` for cells whose centre lies within `radius` of `centre`.
Runs on `hardware.backend`, using `hardware.workgroup` for kernel launch sizing.

Returns the number of cells set.
"""
function setField_Sphere3D!(; mesh, field, value::F, centre::V, radius::F, hardware) where {F <: AbstractFloat, V <: AbstractVector}
    @assert length(centre) == 3 "`centre` must have exactly 3 elements. Use `setField_Circle2D!` for 2-D."
    @assert length(mesh.cells) == length(field.mesh.cells) "`mesh` and `field` must be defined on the same domain"

    (; backend, workgroup) = hardware
    cells   = field.mesh.cells
    ndrange = length(cells)
    c0      = SVector{3,F}(centre[1], centre[2], centre[3])
    matched = KA.zeros(backend, Int64, ndrange)
    kernel! = _setField_Sphere!(_setup(backend, workgroup, ndrange)...)
    kernel!(field, cells, F(value), c0, F(radius), matched)
    KA.synchronize(backend)
    return Int(sum(matched))
end

@kernel function _setField_Sphere!(field, cells, value, centre, radius, matched)
    i = @index(Global)
    @inbounds begin
        c = cells[i].centre
        if norm(c - centre) <= radius
            field[i] = value
            matched[i] = one(Int64)
        end
    end
end

"""
    setField_Expression!(; mesh, field, condition, value_true::F, value_false=nothing, hardware)

Assigns `value_true` to every cell whose centre `(x, y, z)` satisfies `condition(x, y, z)`.
If `value_false` is provided all other cells receive `value_false`; otherwise they are unchanged.
Runs on `hardware.backend` (CPU or GPU), using `hardware.workgroup` for kernel launch sizing.
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
    hardware    = hardware,
)
```
"""
function setField_Expression!(;
    mesh,
    field,
    condition::Cond,
    value_true::F,
    value_false::Union{F,Nothing} = nothing,
    hardware,
) where {F <: AbstractFloat, Cond <: Function}
    @assert length(mesh.cells) == length(field.mesh.cells) "`mesh` and `field` must be defined on the same domain"

    (; backend, workgroup) = hardware
    cells    = field.mesh.cells
    ndrange  = length(cells)
    vf       = value_false !== nothing ? F(value_false) : zero(F)
    has_vf   = value_false !== nothing
    matched  = KA.zeros(backend, Int64, ndrange)
    kernel!  = _setField_Expression!(_setup(backend, workgroup, ndrange)...)
    kernel!(field, cells, condition, F(value_true), vf, has_vf, matched)
    KA.synchronize(backend)
    return Int(sum(matched))
end

@kernel function _setField_Expression!(field, cells, condition::Cond, value_true, value_false, has_vf, matched) where Cond
    i = @index(Global)
    @inbounds begin
        c = cells[i].centre
        if condition(c[1], c[2], c[3])
            field[i] = value_true
            matched[i] = one(Int64)
        elseif has_vf
            field[i] = value_false
        end
    end
end
