# Cell-based gradient limiter (Barth-Jespersen, cf. OpenFOAM cellLimitedGrad): one scalar limiter in [0,1] per cell,
# the minimum over its faces of the ratio keeping the extrapolated face value within neighbour min/max, applied to the whole gradient.
# `level` (0-1) sets limiting strength: 1 = full, 0 = none.

export limit_gradient!
export CellBased

struct CellBased{T<:AbstractFloat}
    level::T
end
Adapt.@adapt_structure CellBased
CellBased() = CellBased(1.0)
function CellBased(level)
    (0 ≤ level ≤ 1) || error("CellBased limiter: `level` must be between 0 and 1, got $level")
    return CellBased{typeof(float(level))}(float(level))
end

limit_gradient!(method::Nothing, ∇F, F, config) = nothing

### GRADIENT LIMITER - EXPERIMENTAL

function limit_gradient!(method::CellBased, ∇F, F::ScalarField, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    mesh = F.mesh
    (; cells, cell_neighbours, cell_faces, faces) = mesh

    (; x, y, z) = ∇F.result

    ndrange = length(cells)
    kernel! = _sized(_limit_gradient!, backend, workgroup, ndrange)
    kernel!(method, x, y, z, F, cells, cell_neighbours, cell_faces, faces)
    sync!(∇F.result, mesh, config) # ghost limiter values are wrong (partial face lists)
end

function limit_gradient!(method::CellBased, ∇F, F::VectorField, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    mesh = F.mesh
    (; cells, cell_neighbours, cell_faces, faces) = mesh

    (; xx, xy, xz) = ∇F.result
    (; yx, yy, yz) = ∇F.result
    (; zx, zy, zz) = ∇F.result

    ndrange = length(cells)
    kernel! = _sized(_limit_gradient!, backend, workgroup, ndrange)
    kernel!(method, xx, xy, xz, F.x, cells, cell_neighbours, cell_faces, faces)
    kernel!(method, yx, yy, yz, F.y, cells, cell_neighbours, cell_faces, faces)
    kernel!(method, zx, zy, zz, F.z, cells, cell_neighbours, cell_faces, faces)
    sync!(∇F.result, mesh, config) # ghost limiter values are wrong (partial face lists)
end

@kernel function _limit_gradient!(method::CellBased, x, y, z, F, cells, cell_neighbours, cell_faces, faces)
    cID = @index(Global)

    @inbounds begin
        faces_range = cells.faces_range[cID]
        phiP = F[cID]
        phiMax = phiP
        phiMin = phiP

        for fi ∈ faces_range
            nID = cell_neighbours[fi]
            phiN = F[nID]
            phiMax = max(phiN, phiMax)
            phiMin = min(phiN, phiMin)
        end

        grad0 = SVector{3}(x[cID], y[cID], z[cID])

        cc = cells.centre[cID]
        limiter = one(phiP)
        ϵ = 10 * eps(phiP)

        for fi ∈ faces_range
            fID = cell_faces[fi]

            fc = faces.centre[fID]
            δϕ = (fc - cc)⋅grad0

            if δϕ > ϵ
                limiter = min(limiter, (phiMax - phiP)/δϕ)
            elseif δϕ < -ϵ
                limiter = min(limiter, (phiMin - phiP)/δϕ)
            end
        end
        limiter = one(phiP) - method.level * (one(phiP) - limiter)
        grad0 *= limiter
        x.values[cID] = grad0[1]
        y.values[cID] = grad0[2]
        z.values[cID] = grad0[3]
    end
end
