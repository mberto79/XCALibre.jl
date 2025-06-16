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
    nboundaries = length(region.boundaries)
    for (name, assignedBC) ∈ zip(names, assignedBCs)
        @assert length(assignedBC) == nboundaries "Inconsistent number of boundaries assigned to field $name"
    end
    return assignedBCs
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







