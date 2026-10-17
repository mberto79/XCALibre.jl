export green_gauss!

# Green gauss for FaceScalarField

function green_gauss!(grad::Grad{S,F,R,I,M}, phif, config) where {S,F,R<:VectorField,I,M}
    (; hardware) = config
    (; backend, workgroup) = hardware

    (; x, y, z) = grad.result
    
    # Launch result calculation kernel
    ndrange = length(x)
    kernel! = _sized(_green_gauss!, backend, workgroup, ndrange)
    kernel!(x, y, z, phif)
    # # KernelAbstractions.synchronize(backend)

    # boundary faces, one work item per boundary cell (fixed summation order)
    ndrange = length(phif.mesh.boundary_cells)
    kernel! = _sized(boundary_faces_contribution!, backend, workgroup, ndrange)
    kernel!(x, y, z, phif)
    # # KernelAbstractions.synchronize(backend)
end

# Green Gauss kernel definition
@kernel function _green_gauss!(dx, dy, dz, phif)
    i = @index(Global)

    @uniform begin
        (; mesh, values) = phif
        (; faces, cells, cell_faces, cell_nsign) = mesh
    end
     
    @inbounds begin
        volume, faces_range = cells.volume[i], cells.faces_range[i]

        z = zero(volume)
        res = SVector{3}(z,z,z)

        for fi ∈ faces_range
            fID = cell_faces[fi]
            nsign = cell_nsign[fi]
            area, normal = faces.area[fID], faces.normal[fID]
            
            res += values[fID]*(area*normal*nsign)
        end
        res /= volume

        dx.values[i] = res[1]
        dy.values[i] = res[2]
        dz.values[i] = res[3]
    end    
end

# Boundary faces contribution kernel definition: one work item per boundary cell adds its faces in
# face order (no atomics), so the result does not depend on the thread count

@kernel function boundary_faces_contribution!(dx, dy, dz, phif)
    j = @index(Global)

    @uniform begin
        (; mesh, values) = phif
        (; faces, cells, boundary_cells, boundary_cell_faces_range, boundary_cell_faces) = mesh
    end

    @inbounds begin
        cID = boundary_cells[j]
        volume = cells.volume[cID]
        for k ∈ boundary_cell_faces_range[j]
            i = boundary_cell_faces[k]
            area, normal = faces.area[i], faces.normal[i]

            res = values[i]*(area*normal)
            res /= volume

            dx.values[cID] += res[1]
            dy.values[cID] += res[2]
            dz.values[cID] += res[3]
        end
    end
end

# Green gauss for FaceVectorField

function green_gauss!(
    grad::Grad{S,F,R,I,M}, psif, config) where {S,F,R<:TensorField,I,M}

    (; hardware) = config
    (; backend, workgroup) = hardware

    (; xx, xy, xz, yx, yy, yz, zx, zy, zz) = grad.result
    
    # Launch result calculation kernel
    ndrange = length(xx)
    kernel! = _sized(_green_gauss_vector!, backend, workgroup, ndrange)
    kernel!(xx, xy, xz, yx, yy, yz, zx, zy, zz, psif)
    # # KernelAbstractions.synchronize(backend)

    # boundary faces, one work item per boundary cell (fixed summation order)
    ndrange = length(psif.mesh.boundary_cells)
    kernel! = _sized(boundary_faces_contribution_vector!, backend, workgroup, ndrange)
    kernel!(xx, xy, xz, yx, yy, yz, zx, zy, zz, psif)
    # # KernelAbstractions.synchronize(backend)
end

# Green Gauss kernel definition
@kernel function _green_gauss_vector!(xx, xy, xz, yx, yy, yz, zx, zy, zz, psif)
    i = @index(Global)

    @uniform begin
        (; mesh) = psif
        (; faces, cells, cell_faces, cell_nsign) = psif.mesh
        z = zero(eltype(xx.values))
    end
     
    @inbounds begin
        volume, faces_range = cells.volume[i], cells.faces_range[i]

        res = SMatrix{3,3}(z,z,z,z,z,z,z,z,z)

        for fi ∈ faces_range
            fID = cell_faces[fi]
            nsign = cell_nsign[fi]
            area, normal = faces.area[fID], faces.normal[fID]
            Sf = area*normal*nsign
            res += psif[fID]*Sf'
        end
        res /= volume

        xx.values[i] = res[1,1]
        xy.values[i] = res[1,2]
        xz.values[i] = res[1,3]

        yx.values[i] = res[2,1]
        yy.values[i] = res[2,2]
        yz.values[i] = res[2,3]

        zx.values[i] = res[3,1]
        zy.values[i] = res[3,2]
        zz.values[i] = res[3,3]
    end    
end

@kernel function boundary_faces_contribution_vector!(
    xx, xy, xz, yx, yy, yz, zx, zy, zz, psif)
    j = @index(Global)

    @uniform begin
        (; mesh) = psif
        (; faces, cells, boundary_cells, boundary_cell_faces_range, boundary_cell_faces) = mesh
    end

    @inbounds begin
        cID = boundary_cells[j]
        volume = cells.volume[cID]
        for k ∈ boundary_cell_faces_range[j]
            i = boundary_cell_faces[k]
            area, normal = faces.area[i], faces.normal[i]

            Sf = area*normal
            res = psif[i]*Sf'
            res /= volume

            xx.values[cID] += res[1,1]
            xy.values[cID] += res[1,2]
            xz.values[cID] += res[1,3]

            yx.values[cID] += res[2,1]
            yy.values[cID] += res[2,2]
            yz.values[cID] += res[2,3]

            zx.values[cID] += res[3,1]
            zy.values[cID] += res[3,2]
            zz.values[cID] += res[3,3]
        end
    end
end
