function delta!(Δ, mesh)
    # Extract hardware configuration
    (; hardware, boundaries) = get_configuration(CONFIG)
    (; backend, workgroup) = hardware

    (; cells) = mesh
    distance, power = delta_scaling(mesh, boundaries[1])

    # set up and launch kernel
    ndrange = length(cells)
    kernel! = _delta!(_setup(backend, workgroup, ndrange)...)
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
        pmin, pmax = bounding_box(mesh)
        if typeof(BC) <: Empty
            n = KernelAbstractions.zeros(CPU(), Float64, 3)
            pmin_cpu = KernelAbstractions.zeros(CPU(), Float64, 3)
            pmax_cpu = KernelAbstractions.zeros(CPU(), Float64, 3)

            copyto!(n, get_normal(mesh, BC)) # copy from device
            copyto!(pmin_cpu, pmin) # copy from device
            copyto!(pmax_cpu, pmax) # copy from device
           
            d1 = norm(pmin_cpu⋅n)
            d2 = norm(pmax_cpu⋅n)
            distance = d1 + d2
            @info "LES Delta: Empty boundary found (ID $(BC.ID)). Treat as 2D"
            @info "LES Delta: Distance between Empty patches $distance"
            return scalar(distance), scalar(0.5) # distance, power
        end
    end
    return scalar(1.0), scalar(1/3) # distance, power
end

function get_normal(mesh, BC)
    (; faces) = mesh
    backend = get_backend(faces)

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
