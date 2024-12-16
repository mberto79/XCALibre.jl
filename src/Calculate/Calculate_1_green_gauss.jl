export green_gauss!

# Green gauss for FaceScalarField

function green_gauss!(grad::Grad{S,F,R,I,M}, phif, config) where {S,F,R<:VectorField,I,M}
    (; hardware) = config
    (; backend, workgroup) = hardware

    (; x, y, z) = grad.result
    
    # Launch result calculation kernel
    kernel! = _green_gauss!(backend, workgroup)
    kernel!(x, y, z, phif, ndrange=length(x))
    KernelAbstractions.synchronize(backend)

    # number of boundary faces
    nbfaces = length(phif.mesh.boundary_cellsID)
    
    kernel! = boundary_faces_contribution!(backend, workgroup)
    kernel!(x, y, z, phif, ndrange=nbfaces)
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
        (; volume, faces_range) = cells[i]

        res = SVector{3}(0.0,0.0,0.0)

        for fi ∈ faces_range
            fID = cell_faces[fi]
            nsign = cell_nsign[fi]
            (; area, normal) = faces[fID]
            
            res += values[fID]*(area*normal*nsign)
        end
        res /= volume

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
        (; ownerCells, area, normal) = faces[i]
        cID = ownerCells[1]
        (; volume) = cells[cID]

        res = values[i]*(area*normal)
        res /= volume 
        
        Atomix.@atomic dx.values[cID] += res[1]
        Atomix.@atomic dy.values[cID] += res[2]
        Atomix.@atomic dz.values[cID] += res[3]
    end
end

# Green gauss for FaceVectorField

function green_gauss!(
    grad::Grad{S,F,R,I,M}, psif, config) where {S,F,R<:TensorField,I,M}

    (; hardware) = config
    (; backend, workgroup) = hardware

    (; xx, xy, xz, yx, yy, yz, zx, zy, zz) = grad.result
    
    # Launch result calculation kernel
    kernel! = _green_gauss_vector!(backend, workgroup)
    kernel!(xx, xy, xz, yx, yy, yz, zx, zy, zz, psif, ndrange=length(xx))
    KernelAbstractions.synchronize(backend)

    # number of boundary faces
    nbfaces = length(psif.mesh.boundary_cellsID)
    
    kernel! = boundary_faces_contribution_vector!(backend, workgroup)
    kernel!(xx, xy, xz, yx, yy, yz, zx, zy, zz, psif, ndrange=nbfaces)
    KernelAbstractions.synchronize(backend)
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
        (; volume, faces_range) = cells[i]

        res = SMatrix{3,3}(z,z,z,z,z,z,z,z,z)

        for fi ∈ faces_range
            fID = cell_faces[fi]
            nsign = cell_nsign[fi]
            (; area, normal) = faces[fID]
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
    i = @index(Global)

    @uniform begin
        (; mesh) = psif
        (; faces, cells) = mesh
    end

    @inbounds begin
        (; ownerCells, area, normal) = faces[i]
        cID = ownerCells[1]
        (; volume) = cells[cID]

        Sf = area*normal
        res = psif[i]*Sf'
        res /= volume 
        
        Atomix.@atomic xx.values[cID] += res[1,1]
        Atomix.@atomic xy.values[cID] += res[1,2]
        Atomix.@atomic xz.values[cID] += res[1,3]

        Atomix.@atomic yx.values[cID] += res[2,1]
        Atomix.@atomic yy.values[cID] += res[2,2]
        Atomix.@atomic yz.values[cID] += res[2,3]

        Atomix.@atomic zx.values[cID] += res[3,1]
        Atomix.@atomic zy.values[cID] += res[3,2]
        Atomix.@atomic zz.values[cID] += res[3,3]
    end
end