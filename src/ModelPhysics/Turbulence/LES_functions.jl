function delta!(Δ, mesh, config)
    # Extract hardware configuration
    (; hardware) = config
    (; backend, workgroup) = hardware

    (; cells) = mesh
    distance, power = delta_scaling(mesh, config.boundaries[1])

    # set up and launch kernel
    ndrange = length(cells)
    kernel! = _sized(_delta!, backend, workgroup, ndrange)
    kernel!(Δ, distance, power, cells)
    # KernelAbstractions.synchronize(backend)
end

@kernel function _delta!(Δ, distance, power, cells) # Add types in call to get float type at compile time
    i = @index(Global)

    @uniform begin
        values = Δ.values
    end

    @inbounds begin
        values[i] = (cells[i].volume/distance)^power
    end
end

delta_scaling(mesh::Mesh2, BCs) = begin # XCALibre's 2D "flat" mesh
    scalar = ScalarFloat(mesh)
    scalar(1.0), scalar(0.5)
end

function delta_scaling(mesh::Mesh3, BCs) # Deal with 2D/3D FOAM grids
    scalar = ScalarFloat(mesh)
    for BC ∈ BCs
        if typeof(BC) <: Empty
            n = SVector{3}(Array(get_normal(mesh, BC))) # copy from device
            distance = extent_along(mesh, n)
            @info "LES Delta: Empty boundary found (ID $(BC.ID)). Treat as 2D"
            @info "LES Delta: Distance between Empty patches $distance"
            return scalar(distance), scalar(0.5) # distance, power
        end
    end
    return scalar(1.0), scalar(1/3) # distance, power
end

# Extent of the mesh nodes along the unit vector n, max(x⋅n) - min(x⋅n): the distance between
# the two Empty planes of a one-cell-thick mesh, wherever the mesh sits relative to the origin
function extent_along(mesh, n::SVector{3,F}) where F
    (; node_coords) = mesh
    lo = mapreduce(x -> x ⋅ n, min, node_coords; init=typemax(F))
    hi = mapreduce(x -> x ⋅ n, max, node_coords; init=typemin(F))
    return hi - lo
end

function get_normal(mesh, BC)
    (; faces) = mesh
    backend = _get_backend(mesh)

    n = KernelAbstractions.zeros(backend, _get_float(mesh), 3)

    kernel! = _get_normal!(backend, 1, 1)
    kernel!(n, faces, BC)
    return n
end

@kernel function _get_normal!(n, faces, BC)
    i = @index(Global)
    ni = faces[BC.IDs_range[1]].normal 
    n[1] = ni[1]
    n[2] = ni[2]
    n[3] = ni[3]
end
