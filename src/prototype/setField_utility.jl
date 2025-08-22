export setField_Box!, setField_Sphere!

using LinearAlgebra


function setField_Box!(mesh, field, value::Float64, region::String)
# example of region: region_to_initialize = "(0 0 -1) (0.05 0.1 1)"

    coords = map(s -> parse(Float64, s), split(replace(region, r"[()]" => ""))) #split input region into array of coords
    p1 = coords[1:3] #min corner of the box
    p2 = coords[4:6] #max corner of the box
    
    min_corner = min.(p1, p2) #account for any order of input coords
    max_corner = max.(p1, p2) #account for any order of input coords

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



function setField_Sphere!(mesh, field, value::Float64, region::String)
    
    # region = "(x y z) r" would imply sphere in 3D
    # region = "(x y) r" would imply circle in 2D

    # THUS we need to check if we have 3 or 4 elements in our region string


    coords = parse.(Float64, split(replace(region, r"[()]" => "")))


    centre, radius = if length(coords) == 4 # 3D sphere
        (coords[1:3], coords[4])
    elseif length(coords) == 3 # 2D circle
        ([coords[1], coords[2], 0.0], coords[3]) # Introduce 0.0 for the third dimension
    else
        error("Invalid region format!!!")
    end


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