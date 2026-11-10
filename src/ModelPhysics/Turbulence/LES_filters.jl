abstract type AbstractFilter end

struct TopHatFilter{F} <: AbstractFilter
    cellArea::F 
end
Adapt.@adapt_structure TopHatFilter

TopHatFilter(field::AbstractField, fieldBCs, config) = begin
    cellArea = cell_surface_area(field, fieldBCs, config)
    TopHatFilter(cellArea)
end

prepass(a, i) = a[i]
postpass(a) = a

accumulator(a::F) where F<:AbstractFloat = zero(F)

accumulator(a::SVector{3,F}) where F = begin
    z = zero(F)
    SVector{3,F}(z, z, z)
end

accumulator(a::SMatrix{3,3,F,9}) where F = begin
    z = zero(F)
    SMatrix{3,3}(z, z, z, z, z, z, z, z, z)
end

(filter::TopHatFilter)(field::AbstractField, cID; pre=prepass, post=postpass) = begin
    @uniform begin
        mesh = _mesh(field)
        (; faces, cells, cell_faces, cell_neighbours, boundary_cellsID) = mesh
        (; cellArea) = filter
    end

    @inbounds begin
        (; volume, faces_range) = cells[cID]

        surfaceSum = accumulator(pre(field, 1)) 

        for fi ∈ faces_range
            fID = cell_faces[fi]
            (; area, weight, ownerCells) = faces[fID]
            cID1 = ownerCells[1] 
            cID2 = ownerCells[2]
            # cIDN = cell_neighbours[fi]
            
            f1 = pre(field, cID1)
            f2 = pre(field, cID2)
            fieldf = weight*f1 + (1 - weight)*f2
            # fieldf = 0.5*f1 + 0.5*f2
            surfaceSum += fieldf*area
        end
        
        # filteredField = post((surfaceSum)/cellArea[cID])
        # return filteredField

        FIDS = findall(isequal(cID), boundary_cellsID)
        for FID ∈ FIDS
            if FID < 561
            face = faces[FID]
            f1 = pre(field, cID)
            surfaceSum += face.area*f1
            end
        end
        filteredField = post(surfaceSum/cellArea[cID])
        return filteredField
    end
end

function cell_surface_area(field, fieldBCs, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    (; mesh) = field

    areaSum = _convert_array!(zeros(_get_float(mesh),length(field)), backend)
    (; boundaries, faces) = mesh
    ndrange=length(field)
    kernel! = _sized(_area_sum!, backend, workgroup, ndrange)
    kernel!(areaSum, mesh)

    # add non-empty boundary contributions
    boundaries_cpu = get_boundaries(boundaries)

    # one work item per boundary cell adds its faces of the patch in face order (deterministic)
    (; boundary_cells, boundary_cell_faces_range, boundary_cell_faces) = mesh
    for BC ∈ fieldBCs
        if typeof(BC) <: Empty
            continue
        end
        facesID_range = boundaries_cpu[BC.ID].IDs_range
        isempty(facesID_range) && continue

        ndrange=length(boundary_cells)
        kernel! = _add_boundary_area!(backend)
        kernel!(areaSum, facesID_range, faces, boundary_cells, boundary_cell_faces_range,
            boundary_cell_faces; _dynamic_setup(backend, workgroup, ndrange)...)
    end
    return areaSum
end

@kernel function _area_sum!(areaSum, mesh)
    i = @index(Global)

    @uniform begin
        (; faces, cells, cell_faces) = mesh
    end
     
    @inbounds begin
        (; faces_range) = cells[i]

        sum = 0.0
        for fi ∈ faces_range
            fID = cell_faces[fi]
            (; area) = faces[fID]
            sum += area
        end
        areaSum[i] = sum
    end
end

@kernel function _add_boundary_area!(
    areaSum, facesID_range, faces, boundary_cells, boundary_cell_faces_range, boundary_cell_faces)
    j = @index(Global)

    @inbounds begin
        cID = boundary_cells[j]
        for k ∈ boundary_cell_faces_range[j]
            fID = boundary_cell_faces[k]
            fID ∈ facesID_range || continue
            areaSum[cID] += faces.area[fID]
        end
    end
end


function basic_filter_new!(phiFiltered, phif, surfaceArea, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    (; boundary_cells) = phiFiltered.mesh
    
    ndrange=length(phiFiltered)
    kernel! = _sized(_surface_sum!, backend, workgroup, ndrange)
    kernel!(phiFiltered, phif, surfaceArea)

    # boundary faces contribution: one work item per boundary cell (deterministic)
    ndrange=length(boundary_cells)
    kernel! = _sized(_add_boundary_faces!, backend, workgroup, ndrange)
    kernel!(phiFiltered, phif, surfaceArea)

end

@kernel function _surface_sum!(phiFiltered, phif::AbstractScalarField, surfaceArea)
    i = @index(Global)

    @uniform begin
        (; mesh) = phiFiltered
        (; faces, cells, cell_faces, cell_nsign) = mesh
    end
     
    @inbounds begin
        (; volume, faces_range) = cells[i]

        # areaSum = 0.0
        # surfaceSum = SVector{3}(0.0,0.0,0.0)
        surfaceSum = 0.0
        for fi ∈ faces_range
            fID = cell_faces[fi]
            (; area) = faces[fID]
            surfaceSum += phif[fID]*area
            # areaSum += area
        end
        res = surfaceSum/surfaceArea[i]

        phiFiltered[i] = res
    end
end

@kernel function _surface_sum!(phiFiltered, phif::AbstractVectorField, surfaceArea)
    i = @index(Global)

    @uniform begin
        (; mesh) = phiFiltered
        (; faces, cells, cell_faces, cell_nsign) = mesh
    end
     
    @inbounds begin
        (; volume, faces_range) = cells[i]

        # areaSum = 0.0
        surfaceSum = SVector{3}(0.0,0.0,0.0)
        # surfaceSum = 0.0
        for fi ∈ faces_range
            fID = cell_faces[fi]
            (; area) = faces[fID]
            surfaceSum += phif[fID]*area
            # areaSum += area
        end
        # res = surfaceSum/areaSum
        res = surfaceSum/surfaceArea[i]

        phiFiltered[i] = res
    end
end

@kernel function _add_boundary_faces!(phiFiltered::Output, phif::Field, surfaceArea) where {Output,Field}
    j = @index(Global)

    @uniform begin
        (; mesh) = phiFiltered
        (; faces, boundary_cells, boundary_cell_faces_range, boundary_cell_faces) = mesh
    end

    cID = boundary_cells[j]
    for k ∈ boundary_cell_faces_range[j]
        fID = boundary_cell_faces[k]
        res = phif[fID]*faces[fID].area/surfaceArea[cID]

        if Output <: AbstractVectorField 
            phiFiltered.x.values[cID] += res[1]
            phiFiltered.y.values[cID] += res[2]
            phiFiltered.z.values[cID] += res[3]
        elseif Output <: AbstractScalarField
            phiFiltered.values[cID] += res
        end
    end
end



function basic_filter!(phiFiltered, phi, surfaceArea, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    # (; x, y, z) = grad.result
    
    # # Launch result calculation kernel
    ndrange=length(phiFiltered)
    kernel! = _sized(_integrate_surface!, backend, workgroup, ndrange)
    kernel!(phiFiltered, phi, surfaceArea)
    KernelAbstractions.synchronize(backend)

end

@kernel function _integrate_surface!(phiFiltered, phi::ScalarField, surfaceArea)
    i = @index(Global)

    @uniform begin
        # (; mesh, values) = phif
        (; mesh) = phi
        (; faces, cells, cell_faces, cell_nsign) = mesh
    end
     
    @inbounds begin
        (; volume, faces_range) = cells[i]

        # areaSum = 0.0
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
            # areaSum += area
        end
        phiFiltered[i] = surfaceSum/surfaceArea[i]
    end
end

@kernel function _integrate_surface!(phiFiltered, phi::VectorField, surfaceArea)
    i = @index(Global)

    @uniform begin
        # (; mesh, values) = phif
        (; mesh) = phi
        (; faces, cells, cell_faces, cell_nsign) = mesh
    end
     
    @inbounds begin
        (; volume, faces_range) = cells[i]

        # areaSum = 0.0
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
            # areaSum += area
        end
        phiFiltered[i] = surfaceSum/surfaceArea[i]
    end
end

@kernel function _integrate_surface!(phiFiltered, phi::AbstractTensorField, surfaceArea)
    i = @index(Global)

    @uniform begin
        # (; mesh, values) = phif
        (; mesh) = phiFiltered
        (; faces, cells, cell_faces, cell_nsign) = mesh
    end
     
    @inbounds begin
        (; volume, faces_range) = cells[i]

        # areaSum = 0.0
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
            # areaSum += area
        end
        phiFiltered[i] = surfaceSum/surfaceArea[i]
    end
end