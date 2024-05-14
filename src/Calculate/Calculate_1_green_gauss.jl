export green_gauss!

# Green gauss function definition

function green_gauss!(dx, dy, dz, phif, config)
    # Retrieve required varaibles for function
    (; mesh, values) = phif
    (; faces, cells, cell_faces, cell_nsign) = mesh
    # backend = _get_backend(mesh)
    (; hardware) = config
    (; backend, workgroup) = hardware

    # Retrieve user-selected float type
    F = _get_float(mesh)
    
    # Launch result calculation kernel
    kernel! = result_calculation!(backend, workgroup)
    kernel!(values, faces, cells, cell_nsign, cell_faces, F, dx, dy, dz, ndrange = length(cells))
    KernelAbstractions.synchronize(backend)

    # Retrieve number of boundary faces
    nbfaces = length(mesh.boundary_cellsID)
    
    # Launch boundary faces contribution kernel
    kernel! = boundary_faces_contribution!(backend, workgroup)
    kernel!(values, faces, cells, F, dx, dy, dz, ndrange = nbfaces)
    KernelAbstractions.synchronize(backend)
end

# Result calculation kernel definition

@kernel function result_calculation!(values, faces, cells, cell_nsign, cell_faces, F, dx, dy, dz)
    i = @index(Global)

    @inbounds begin
        # Extract required fields from work item cell
        (; volume, faces_range) = cells[i]

        # Allocate static vector for results
        res = SVector{3}(0.0,0.0,0.0)

        # Loop over cell faces to calculate result
        for fi ∈ faces_range
            # Define reqquired variables
            fID = cell_faces[fi]
            nsign = cell_nsign[fi]
            (; area, normal) = faces[fID]
            
            # Result calcaultion
            res += values[fID]*(area*normal*nsign)
        end
        # Normalise results with cell volume
        res /= volume

        # Store results in x, y, z value array
        dx.values[i] = res[1]
        dy.values[i] = res[2]
        dz.values[i] = res[3]
    end    
end

# Boundary faces contribution kernel definition

@kernel function boundary_faces_contribution!(values, faces, cells, F, dx, dy, dz)
    i = @index(Global)

    @inbounds begin
        # Extract required variables from work item face and cell
        (; ownerCells, area, normal) = faces[i]
        cID = ownerCells[1]
        (; volume) = cells[cID]

        # Allocate static vector for results
        res = SVector{3}(0.0,0.0,0.0)

        # Results calculation
        res = values[i]*(area*normal)

        # Normalise results with cell volume
        res /= volume 
        
        # Increment x, y, z values vectors with results
        Atomix.@atomic dx.values[cID] += res[1]
        Atomix.@atomic dy.values[cID] += res[2]
        Atomix.@atomic dz.values[cID] += res[3]
    end
end