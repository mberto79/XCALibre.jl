export green_gauss!

# function green_gauss!(grad::Grad, phif; source=false)
# function green_gauss!(dx, dy, dz, phif)
#     # (; x, y, z) = grad.result
#     (; mesh, values) = phif
#     # (; cells, faces) = mesh
#     (; faces, cells, cell_faces, cell_nsign) = mesh
#     F = _get_float(mesh)
#     for ci ∈ eachindex(cells)
#         # (; facesID, nsign, volume) = cells[ci]
#         cell = cells[ci]
#         (; volume) = cell
#         res = SVector{3,F}(0.0,0.0,0.0)
#         # for fi ∈ eachindex(facesID)
#         for fi ∈ cell.faces_range
#             # fID = facesID[fi]
#             fID = cell_faces[fi]
#             nsign = cell_nsign[fi]
#             (; area, normal) = faces[fID]
#             # res += values[fID]*(area*normal*nsign[fi])
#             res += values[fID]*(area*normal*nsign)
#         end
#         res /= volume
#         dx[ci] = res[1]
#         dy[ci] = res[2]
#         dz[ci] = res[3]
#     end
#     # Add boundary faces contribution
#     nbfaces = length(mesh.boundary_cellsID)
#     for i ∈ 1:nbfaces
#         face = faces[i]
#         (; ownerCells, area, normal) = face
#         cID = ownerCells[1] 
#         (; volume) = cells[cID]
#         res = values[i]*(area*normal)
#         res /= volume
#         dx[cID] += res[1]
#         dy[cID] += res[2]
#         dz[cID] += res[3]
#     end
# end

function green_gauss!(dx, dy, dz, phif)
    # (; x, y, z) = grad.result
    (; mesh, values) = phif
    # (; cells, faces) = mesh
    (; faces, cells, cell_faces, cell_nsign) = mesh
    F = _get_float(mesh)

    backend = _get_backend(mesh)
    
    kernel! = result_calculation!(backend, 2)
    kernel!(values, faces, cells, cell_nsign, cell_faces, F, dx, dy, dz, ndrange = length(cells))
    KernelAbstractions.synchronize(backend)

    nbfaces = length(mesh.boundary_cellsID)
    
    kernel! = boundary_faces_contribution!(backend, 2)
    kernel!(values, faces, cells, F, dx, dy, dz, ndrange = nbfaces)
    KernelAbstractions.synchronize(backend)
end

@kernel function result_calculation!(values, faces, cells, cell_nsign, cell_faces, F, dx, dy, dz)
    i = @index(Global)

    @inbounds begin
        (; volume, faces_range) = cells[i]

        # for fi in faces_range
        #     fID = cell_faces[fi]
        #     nsign = cell_nsign[fi]
        #     (; area, normal) = faces[fID]
        #     dx[i] += values[fID]*(area*normal[1]*nsign)
        #     dy[i] += values[fID]*(area*normal[2]*nsign)
        #     dz[i] += values[fID]*(area*normal[3]*nsign)
        # end
        # dx[i] /= volume
        # dy[i] /= volume
        # dz[i] /= volume
        # res = SVector{3,F}(0.0,0.0,0.0)
        res = SVector{3}(0.0,0.0,0.0)

        for fi ∈ faces_range
            # fID = facesID[fi]
            fID = cell_faces[fi]
            nsign = cell_nsign[fi]
            (; area, normal) = faces[fID]
            # res += values[fID]*(area*normal*nsign[fi])
            res += values[fID]*(area*normal*nsign)
        end
        res /= volume
        dx.values[i] = res[1]
        dy.values[i] = res[2]
        dz.values[i] = res[3]
    end    
end

@kernel function boundary_faces_contribution!(values, faces, cells, F, dx, dy, dz)
    i = @index(Global)

    @inbounds begin
        (; ownerCells, area, normal) = faces[i]
        cID = ownerCells[1]
        (; volume) = cells[cID]
        # Atomix.@atomic dx.values[cID] += (values[i]*(area*normal[1]))/volume
        # Atomix.@atomic dy.values[cID] += (values[i]*(area*normal[2]))/volume
        # Atomix.@atomic dz.values[cID] += (values[i]*(area*normal[3]))/volume

        # res = SVector{3,F}(0.0,0.0,0.0)
        res = SVector{3}(0.0,0.0,0.0)
        res = values[i]*(area*normal)
        res /= volume 
        Atomix.@atomic dx.values[cID] += res[1]
        Atomix.@atomic dy.values[cID] += res[2]
        Atomix.@atomic dz.values[cID] += res[3]
    end
end