export limit_gradient!
export FaceBased

struct FaceBased end

### GRADIENT LIMITER - EXPERIMENTAL

function limit_gradient!(method::FaceBased, ∇F, F::ScalarField, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    (; cells, faces, boundary_cellsID,) = F.mesh

    limiter = fill!(allocate(backend, eltype(F), length(cells)), one(eltype(F)))

    nbfaces = length(boundary_cellsID)
    internal_faces = length(faces) - nbfaces

    kernel! = _limit_gradient!(backend, workgroup)
    kernel!(method, limiter, ∇F, F, cells, faces, nbfaces, ndrange=internal_faces)
    KernelAbstractions.synchronize(backend)

    # ∇F.result .*= limiter
    limiter
end

# function limit_gradient!(method::FaceBased, ∇F, F::VectorField, config)
#     (; hardware) = config
#     (; backend, workgroup) = hardware

#     mesh = F.mesh
#     (; cells, cell_neighbours, cell_faces, cell_nsign, faces) = mesh

#     (; xx, yx, zx) = ∇F.result
#     (; xy, yy, zy) = ∇F.result
#     (; xz, yz, zz) = ∇F.result

#     kernel! = _limit_gradient!(backend, workgroup)
#     kernel!(method, xx, yx, zx, F.x, cells, cell_neighbours, cell_faces, cell_nsign, faces, ndrange=length(cells))
#     KernelAbstractions.synchronize(backend)

#     kernel! = _limit_gradient!(backend, workgroup)
#     kernel!(method, xy, yy, zy, F.y, cells, cell_neighbours, cell_faces, cell_nsign, faces, ndrange=length(cells))
#     KernelAbstractions.synchronize(backend)

#     kernel! = _limit_gradient!(backend, workgroup)
#     kernel!(method, xz, yz, zz, F.z, cells, cell_neighbours, cell_faces, cell_nsign, faces, ndrange=length(cells))
#     KernelAbstractions.synchronize(backend)
# end

@kernel function _limit_gradient!(::FaceBased, limiter, ∇F, F, cells, faces, nbfaces)
    i = @index(Global)
    fID = i + nbfaces

    face = faces[fID]
    ownerCells = face.ownerCells
    owner1 = ownerCells[1]
    owner2 = ownerCells[2]
    cell1 = cells[owner1]
    cell2 = cells[owner2]

    n = face.normal
    cf = face.centre 
    c1 = cell1.centre
    c2 = cell2.centre
    d1 = (cf - c1)
    d2 = (cf - c2)


    F1 = F[owner1]
    F2 = F[owner2]
    grad1 = ∇F[owner1]
    grad2 = ∇F[owner2]

    minF = min(F1, F2)
    maxF = max(F1, F2)
    deltaF = (maxF - minF)
    minF -= deltaF
    maxF += deltaF
    F1_ext = d1⋅grad1
    F2_ext = d2⋅grad2

    set_limiter(limiter, owner1, maxF - F1, minF - F1, F1_ext)
    set_limiter(limiter, owner2, maxF - F2, minF - F2, F2_ext)
end

function set_limiter(limiter, cID, δmax, δmin, δF)
    # if δF > 0 && δmax > 0 && δF > δmax
    if δF > δmax
        limiter[cID] = min(limiter[cID], δmax/δF)
    # elseif δF < 0 && δmin < 0 && δF < δmin
    elseif δF < δmin
        limiter[cID] = min(limiter[cID], δmin/δF)
    end
end  