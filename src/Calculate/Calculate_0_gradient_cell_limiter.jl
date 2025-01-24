# THIS IMPLEMENTATION IS WORK IN PROGRESS AND HAS NOT BEEN TESTED FOR CORRECTNESS

export limit_gradient!
export CellBased

struct CellBased end

limit_gradient!(method::Nothing, ∇F, F, config) = nothing

### GRADIENT LIMITER - EXPERIMENTAL

function limit_gradient!(method::CellBased, ∇F, F::ScalarField, config)
# function limit_gradient!(∇F, Ff, F::ScalarField, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    mesh = F.mesh
    (; cells, cell_neighbours, cell_faces, cell_nsign, faces) = mesh

    (; x, y, z) = ∇F.result

    kernel! = _limit_gradient!(backend, workgroup)
    # kernel!(x, y, z, Ff, F, cells, cell_neighbours, cell_faces, cell_nsign, faces, ndrange=length(cells))
    kernel!(method, x, y, z, F, cells, cell_neighbours, cell_faces, cell_nsign, faces, ndrange=length(cells))
    KernelAbstractions.synchronize(backend)
end

function limit_gradient!(method::CellBased, ∇F, F::VectorField, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    mesh = F.mesh
    (; cells, cell_neighbours, cell_faces, cell_nsign, faces) = mesh

    (; xx, yx, zx) = ∇F.result
    (; xy, yy, zy) = ∇F.result
    (; xz, yz, zz) = ∇F.result

    kernel! = _limit_gradient!(backend, workgroup)
    kernel!(method, xx, yx, zx, F.x, cells, cell_neighbours, cell_faces, cell_nsign, faces, ndrange=length(cells))
    KernelAbstractions.synchronize(backend)

    kernel! = _limit_gradient!(backend, workgroup)
    kernel!(method, xy, yy, zy, F.y, cells, cell_neighbours, cell_faces, cell_nsign, faces, ndrange=length(cells))
    KernelAbstractions.synchronize(backend)

    kernel! = _limit_gradient!(backend, workgroup)
    kernel!(method, xz, yz, zz, F.z, cells, cell_neighbours, cell_faces, cell_nsign, faces, ndrange=length(cells))
    KernelAbstractions.synchronize(backend)
end

# @kernel function _limit_gradient!(x, y, z, Ff, F, cells, cell_neighbours, cell_faces, cell_nsign, faces)
@kernel function _limit_gradient!(::CellBased, x, y, z, F, cells, cell_neighbours, cell_faces, cell_nsign, faces)
    cID = @index(Global)

    cell = cells[cID]
    faces_range = cell.faces_range
    phiP = F[cID]
    phiMax = phiMin = phiP
 
    for fi ∈ faces_range
        nID = cell_neighbours[fi]
        phiN = F[nID]
        
        # fID = cell_faces[fi]
        # phiN = Ff[fID]

        phiMax = max(phiN, phiMax)
        phiMin = min(phiN, phiMin)
    end

    # g0 = ∇F[cID]
    grad0 = SVector{3}(x[cID] , y[cID] , z[cID])

    cc = cell.centre
    uno = one(eltype(F[cID]))
    limiter = uno
    limiterf = uno
    for fi ∈ faces_range 
        fID = cell_faces[fi]
        nID = cell_neighbours[fi]
        face = faces[fID]
        cellN = cells[nID]
        # nID = face.ownerCells[2]
        # phiN = F[nID]
        normal = face.normal
        nsign = cell_nsign[fi]
        na = nsign*normal

        
        fc = face.centre
        nc = cellN.centre
        δϕ = (nc - cc)⋅grad0
        # δϕ = (fc - cc)⋅grad0

        if δϕ > 0
            limiterf = min(limiter, (phiMax - phiP)/δϕ)
        elseif δϕ < 0
            limiterf = min(limiter, (phiMin - phiP)/δϕ)
        # else
        #     limiterf = uno
        end
        # limiter = min(limiterf, limiter)
        limiter = limiterf
    end
    grad0 *= limiter
    x.values[cID] = grad0[1]
    y.values[cID] = grad0[2]
    z.values[cID] = grad0[3]
end