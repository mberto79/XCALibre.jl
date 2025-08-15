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
    kernel! = _area_sum!(_setup(backend, workgroup, ndrange)...)
    kernel!(areaSum, mesh)

    # add non-empty boundary contributions
    boundaries_cpu = get_boundaries(boundaries)

    for BC ∈ fieldBCs
        if typeof(BC) <: Empty
            continue
        end
        facesID_range = boundaries_cpu[BC.ID].IDs_range
        start_ID = facesID_range[1]
        
        kernel_range = length(facesID_range)

        ndrange=kernel_range
        kernel! = _add_boundary_area!(_setup(backend, workgroup, ndrange)...)
        kernel!(areaSum, start_ID, faces)
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

@kernel function _add_boundary_area!(areaSum, start_ID, faces)
    i = @index(Global)
    fID = i + start_ID - 1

    @inbounds begin
        face = faces[fID]
        cID = face.ownerCells[1]
        Atomix.@atomic areaSum[cID] += face.area
    end
end


function basic_filter_new!(phiFiltered, phif, surfaceArea, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    (; boundary_cellsID) = phiFiltered.mesh
    
    ndrange=length(phiFiltered)
    kernel! = _surface_sum!(_setup(backend, workgroup, ndrange)...)
    kernel!(phiFiltered, phif, surfaceArea)

    # boundary faces contribution 
    ndrange=length(boundary_cellsID)
    kernel! = _add_boundary_faces!(_setup(backend, workgroup, ndrange)...)
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
    fID = @index(Global)

    @uniform begin
        (; mesh) = phiFiltered
        (; faces, cells, cell_faces, boundary_cellsID) = mesh
    end

    cID = boundary_cellsID[fID]
    (; faces_range) = cells[cID]

    # areaSum = 0.0
    surfaceSum = nothing 
    if Field <: AbstractVectorField 
        surfaceSum = SVector{3}(0.0,0.0,0.0)
    elseif Field <: AbstractScalarField
        surfaceSum = 0.0
    end 

    # area = nothing
    # for fi ∈ faces_range
    #     cfID = cell_faces[fi] # cell-based face ID
    #     (; area) = faces[cfID]
    #     # surfaceSum += phif[cfID]*area
    #     # areaSum += area
    # end
    bfarea = faces[fID].area
    # areaSum += bfarea
    # surfaceSum += phif[fID]*bfarea
    # res = surfaceSum/areaSum
    surfaceSum = phif[fID]*bfarea
    res = surfaceSum/surfaceArea[cID]

    # Atomix.@atomic phiFiltered[cID] += res

    if Output <: AbstractVectorField 
        Atomix.@atomic phiFiltered.x.values[cID] += res[1]
        Atomix.@atomic phiFiltered.y.values[cID] += res[2]
        Atomix.@atomic phiFiltered.z.values[cID] += res[3]
    elseif Output <: AbstractScalarField
        Atomix.@atomic phiFiltered.values[cID] += res
    end 
end



function basic_filter!(phiFiltered, phi, surfaceArea, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    # (; x, y, z) = grad.result
    
    # # Launch result calculation kernel
    ndrange=length(phiFiltered)
    kernel! = _integrate_surface!(_setup(backend, workgroup, ndrange)...)
    kernel!(phiFiltered, phi, surfaceArea)
    KernelAbstractions.synchronize(backend)

    # # number of boundary faces
    # nbfaces = length(phif.mesh.boundary_cellsID)
    
    # ndrange=nbfaces
    # kernel! = boundary_faces_contribution!(_setup(backend, workgroup, ndrange)...)
    # kernel!(x, y, z, phif)
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
            # isowner = signbit(-nsign) # owner if nsign is positive - so negating 
            # notowner = signbit(nsign) # not owner if nsign is positive
            # w = 1*notowner - weight*notowner + weight*isowner # correct if not owner
            # oneMinusW = 1 - w
            # phif = phi[cID1]*w + phi[cID2]*oneMinusW

            # phif = phi[cID1]*weight + phi[cID2]*(1 - weight)
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
            # isowner = signbit(-nsign) # owner if nsign is positive - so negating 
            # notowner = signbit(nsign) # not owner if nsign is positive
            # w = 1*notowner - weight*notowner + weight*isowner # correct if not owner
            # oneMinusW = 1 - w
            # phif = phi[cID1]*w + phi[cID2]*oneMinusW

            # phif = phi[cID1]*weight + phi[cID2]*(1 - weight)
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
            # isowner = signbit(-nsign) # owner if nsign is positive - so negating 
            # notowner = signbit(nsign) # not owner if nsign is positive
            # w = 1*notowner - weight*notowner + weight*isowner # correct if not owner
            # oneMinusW = 1 - w
            # phif = phi[cID1]*w + phi[cID2]*oneMinusW

            # phif = phi[cID1]*weight + phi[cID2]*(1 - weight)
            phif = phi[cID1]*0.5 + phi[cID2]*0.5
            surfaceSum += phif*area
            # areaSum += area
        end
        phiFiltered[i] = surfaceSum/surfaceArea[i]
    end
end