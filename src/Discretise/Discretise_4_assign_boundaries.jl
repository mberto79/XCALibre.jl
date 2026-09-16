export assign

(::Type{T})(name::Symbol, value) where T<:AbstractBoundary = T(name,value,0:0)

function assign(args; region)
    BCs = []
    names = propertynames(args)
    for arg ∈ args
        updatedBCs = assign_patches(arg, region)
        push!(BCs, updatedBCs)
    end
    assignedBCs = NamedTuple{names}(Tuple.(BCs))
    boundaries = get_boundaries(region.boundaries)
    for (name, assignedBC) ∈ zip(names, assignedBCs)
        validate_boundary_coverage(name, assignedBC, boundaries)
    end
    return assignedBCs
end

function validate_boundary_coverage(field_name, assigned_bcs, boundaries)
    counts = zeros(Int, length(boundaries))
    for bc in assigned_bcs
        counts[Int(bc.ID)] += 1
    end

    unassigned = [boundaries[id].name for id in eachindex(boundaries) if counts[id] == 0]
    repeated = [boundaries[id].name for id in eachindex(boundaries) if counts[id] > 1]
    isempty(unassigned) && isempty(repeated) && return nothing

    details = String[]
    isempty(unassigned) || push!(details, "missing $(join(unassigned, ", "))")
    isempty(repeated) || push!(details, "assigned more than once $(join(repeated, ", "))")
    throw(ArgumentError(
        "incomplete boundary assignment for field '$field_name': $(join(details, "; "))",
    ))
end

function assign_patches(BCs, region)
    newBCs = []
    for (i, BC) ∈ enumerate(BCs)
        ID, IDs_range = patch_and_faces_IDs(BC, region)
        value = adapt_value(BC.value, region)
        push!(newBCs, typeof(BC).name.wrapper(ID, value, IDs_range))
    end
    # Tuple(newBCs)
    newBCs
end

function patch_and_faces_IDs(BC, mesh)
    # (; boundaries) = mesh # needs to be a copy
    boundaries_cpu = get_boundaries(mesh.boundaries)
    intType = _get_int(mesh)
    for (ID, boundary) ∈ enumerate(boundaries_cpu)
        if BC.ID == boundary.name
            return intType(ID), boundary.IDs_range
        end
    end
    error(""""$(BC.ID)" is not a recognised boundary name""")
end

adapt_value(value::Number, mesh) = _get_float(mesh)(value)
adapt_value(value::Vector, mesh) = begin
    F = _get_float(mesh)
    @assert length(value) == 3 "Vectors must have 3 components"
    SVector{3,F}(value)
end
adapt_value(value::Function, mesh) = value
