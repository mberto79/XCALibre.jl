export setField_Box!, setField_Circle2D!, setField_Sphere3D!, setField_Expression!

"""
    setField_Box!(; mesh, field, value::F, min_corner::V, max_corner::V) where {F <: AbstractFloat, V <: AbstractVector}

Sets field values equal to `value` argument only for cells located within a box defined by min and max corners

# Input arguments

- `mesh` reference to `domain` inside a `Physics` model defined by the user.
- `field` field to be modified.
- `value` new values for the chosen field.
- `min_corner` minimum coordinate values of the box in the format of [x, y, z].
- `max_corner` maximum coordinate values of the box in the format of [x, y, z].
"""
function setField_Box!(; mesh, field, value::F, min_corner::V, max_corner::V) where {F <: AbstractFloat, V <: AbstractVector}

    @assert length(min_corner) == 3 "`min_corner` must have exactly 3 elements"
    @assert length(max_corner) == 3 "`max_corner` must have exactly 3 elements"

    cells_in_region = Int[]
    
    for (id, cell) in enumerate(mesh.cells) #check that X, Y, Z coords are within the box
        center = cell.centre
        if (min_corner[1] <= center[1] <= max_corner[1] &&
            min_corner[2] <= center[2] <= max_corner[2] &&
            min_corner[3] <= center[3] <= max_corner[3])
            
            push!(cells_in_region, id)
            # Warning: if outer cell boundary is outside the box but its centre is within the box, this cell will count too
        end
    end

    if !isempty(cells_in_region)
        field.values[cells_in_region] .= value # Reassign values inside the field
    end
    
    return length(cells_in_region)
end




"""
    setField_Circle2D!(mesh, field, value::F, centre::V, radius::F) where {F <: AbstractFloat, V <: AbstractVector}

Sets field values equal to `value` argument only for cells located within a 2D circle defined by its centre and radius

# Input arguments

- `mesh` reference to `domain` inside a `Physics` model defined by the user.
- `field` field to be modified.
- `value` new values for the chosen field.
- `centre` coordinate of centre of the circle in the format of [x, y].
- `radius` length of the radius of the circle.
"""
function setField_Circle2D!(; mesh, field, value::F, centre::V, radius::F) where {F <: AbstractFloat, V <: AbstractVector}
    
    @assert length(centre) == 2 "`centre` must have exactly 2 elements. Please, use `setField_Sphere3D!` if you need a sphere."
    
    # WARNING : Currently works similar to 2D UNV format e.g. supports meshes in the X-Y plane by default

    centre = [centre..., zero(F)]

    cells_in_region = Int[]

    for (id, cell) in enumerate(mesh.cells) # Loop over cells and select those that are inside desired radius
        if norm(cell.centre .- centre) <= radius # Check if the cell centre is less than 1 radius away from centre coord
            push!(cells_in_region, id)
        end
    end

    if !isempty(cells_in_region)
        field.values[cells_in_region] .= value # Reassign values inside the field
    end

    return length(cells_in_region)
end



"""
    setField_Sphere3D!(mesh, field, value::F, centre::V, radius::F) where {F <: AbstractFloat, V <: AbstractVector}

Sets field values equal to `value` argument only for cells located within a 3D sphere defined by its centre and radius

# Input arguments

- `mesh` reference to `domain` inside a `Physics` model defined by the user.
- `field` field to be modified.
- `value` new values for the chosen field.
- `centre` coordinate of centre of the sphere in the format of [x, y, z].
- `radius` length of the radius of the sphere.
"""
function setField_Sphere3D!(; mesh, field, value::F, centre::V, radius::F) where {F <: AbstractFloat, V <: AbstractVector}
    @assert length(centre) == 3 "centre must have exactly 3 elements. Please, use `setField_Circle2D!` if you need a circle."

    cells_in_region = Int[]

    for (id, cell) in enumerate(mesh.cells) # Loop over cells and select those that are inside desired radius
        if norm(cell.centre .- centre) <= radius # Check if the cell centre is less than 1 radius away from centre coord
            push!(cells_in_region, id)
        end
    end

    if !isempty(cells_in_region)
        field.values[cells_in_region] .= value # Reassign values inside the field
    end

    return length(cells_in_region)
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