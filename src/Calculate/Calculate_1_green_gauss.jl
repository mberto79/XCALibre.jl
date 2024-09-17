export green_gauss!

# Green gauss function definition

function green_gauss!(dx, dy, dz, phif, config)
    # Retrieve required varaibles for function
    (; hardware) = config
    (; backend, workgroup) = hardware
    
    # Launch result calculation kernel
    kernel! = _green_gauss!(backend, workgroup)
    kernel!(dx, dy, dz, phif, ndrange=length(dx))
    KernelAbstractions.synchronize(backend)

    # Retrieve number of boundary faces
    nbfaces = length(phif.mesh.boundary_cellsID)
    
    # Launch boundary faces contribution kernel
    kernel! = boundary_faces_contribution!(backend, workgroup)
    kernel!(dx, dy, dz, phif, ndrange=nbfaces)
    KernelAbstractions.synchronize(backend)
end

# Green Gauss kernel definition
@kernel function _green_gauss!(dx, dy, dz, phif)
    i = @index(Global)

    @uniform begin
        (; mesh, values) = phif
        (; faces, cells, cell_faces, cell_nsign) = mesh
    end
     
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

@kernel function boundary_faces_contribution!(dx, dy, dz, phif)
    i = @index(Global)

    @uniform begin
        (; mesh, values) = phif
        (; faces, cells) = mesh
    end

    @inbounds begin
        # Extract required variables from work item face and cell
        (; ownerCells, area, normal) = faces[i]
        cID = ownerCells[1]
        (; volume) = cells[cID]

        # Allocate static vector for results
        # res = SVector{3}(0.0,0.0,0.0)

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