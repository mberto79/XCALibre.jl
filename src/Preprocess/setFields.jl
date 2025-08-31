using LinearAlgebra

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
