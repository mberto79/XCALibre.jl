function basic_filter!(phiFiltered, phi, config)
    # interpolate!(phif, phi, config)   
    # correct_boundaries!(phif, phi, phi.BCs, time, config)
    # integrate_surface!(phiFiltered, phif, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    # (; x, y, z) = grad.result
    
    # # Launch result calculation kernel
    kernel! = _integrate_surface!(backend, workgroup)
    kernel!(phiFiltered, phi, ndrange=length(phiFiltered))

end

@kernel function _integrate_surface!(phiFiltered, phi::ScalarField)
    i = @index(Global)

    @uniform begin
        # (; mesh, values) = phif
        (; mesh) = phi
        (; faces, cells, cell_faces, cell_nsign) = mesh
    end
     
    @inbounds begin
        (; volume, faces_range) = cells[i]

        areaSum = 0.0
        # surfaceSum = SVector{3}(0.0,0.0,0.0)
        surfaceSum = 0.0
        for fi ∈ faces_range
            fID = cell_faces[fi]
            nsign = cell_nsign[fi]
            (; area, weight, ownerCells) = faces[fID]
            cID1 = ownerCells[1]
            cID2 = ownerCells[2]
            phif = phi[cID1]*0.5 + phi[cID2]*0.5
            surfaceSum += phif*area
            areaSum += area
        end
        res = surfaceSum/areaSum

        phiFiltered[i] = res
    end
end

@kernel function _integrate_surface!(phiFiltered, phi::VectorField)
    i = @index(Global)

    @uniform begin
        # (; mesh, values) = phif
        (; mesh) = phi
        (; faces, cells, cell_faces, cell_nsign) = mesh
    end
     
    @inbounds begin
        (; volume, faces_range) = cells[i]

        areaSum = 0.0
        surfaceSum = SVector{3}(0.0,0.0,0.0)
        # surfaceSum = 0.0
        for fi ∈ faces_range
            fID = cell_faces[fi]
            nsign = cell_nsign[fi]
            (; area, weight, ownerCells) = faces[fID]
            cID1 = ownerCells[1]
            cID2 = ownerCells[2]
            phif = phi[cID1]*0.5 + phi[cID2]*0.5
            surfaceSum += phif*area
            areaSum += area
        end
        res = surfaceSum/areaSum

        phiFiltered[i] = res
    end
end

@kernel function _integrate_surface!(phiFiltered, phi::AbstractTensorField)
    i = @index(Global)

    @uniform begin
        # (; mesh, values) = phif
        (; mesh) = phiFiltered
        (; faces, cells, cell_faces, cell_nsign) = mesh
    end
     
    @inbounds begin
        (; volume, faces_range) = cells[i]

        areaSum = 0.0
        surfaceSum = SMatrix{3,3}(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        # surfaceSum = 0.0
        for fi ∈ faces_range
            fID = cell_faces[fi]
            nsign = cell_nsign[fi]
            (; area, weight, ownerCells) = faces[fID]
            cID1 = ownerCells[1]
            cID2 = ownerCells[2]
            phif = phi[cID1]*0.5 + phi[cID2]*0.5
            surfaceSum += phif*area
            areaSum += area
        end
        res = surfaceSum/areaSum

        phiFiltered[i] = res
    end
end