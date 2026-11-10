# Same neighbourhood bounds as CellBased (Barth-Jespersen), with the smooth per-face function of Venkatakrishnan
# (1995) in place of min(1, ratio); ε² = (K h)³, h = volume^(1/3) (area^(1/2) on 2D meshes).

export Venkatakrishnan

"""
    Venkatakrishnan(K=5.0)

Cell-based gradient limiter with the smooth limiter function of Venkatakrishnan (1995), selected with
`Schemes(limiter=Venkatakrishnan(K))`. Like `CellBased` it scales the cell gradient by one factor in
[0, 1] so that the values extrapolated to the faces stay within the minimum and maximum of the cell and its
face neighbours, but the factor is a smooth function of the ratio of allowed to extrapolated change instead of
`min(1, ratio)`. Limiting therefore starts gradually (the factor is 1 only from a ratio of 2) and does not
switch on and off from one iteration to the next.

`K ≥ 0` controls the threshold ε² = (K h)³ below which differences are not limited, with `h` the cell length
scale (cube root of the volume, square root of the area on 2D meshes). ε² is compared with squared differences of
the limited field, so its effect depends on the units of the field: for velocities and pressures in SI units
and cell sizes well below 1 m it is negligible unless `K` is large. `K = 0` limits all differences.
"""
struct Venkatakrishnan{T<:AbstractFloat}
    K::T
    function Venkatakrishnan(K::Real)
        (K ≥ 0) || error("Venkatakrishnan limiter: `K` must be non-negative, got $K")
        return new{typeof(float(K))}(float(K))
    end
end
Adapt.@adapt_structure Venkatakrishnan
Venkatakrishnan() = Venkatakrishnan(5.0)

# exponent turning the cell measure into a length (2D meshes store the cell area as volume)
_cell_length_power(::Mesh2) = 1//2
_cell_length_power(::Mesh3) = 1//3
_cell_length_power(mesh::AbstractMesh) = _cell_length_power(mesh.mesh) # wrapped meshes (DistributedMesh)

function limit_gradient!(method::Venkatakrishnan, ∇F, F::ScalarField, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    mesh = F.mesh
    (; cells, cell_neighbours, cell_faces, faces) = mesh
    TF = _get_float(mesh)
    K = TF(method.K)
    power = TF(_cell_length_power(mesh))

    (; x, y, z) = ∇F.result

    ndrange = length(cells)
    kernel! = _sized(_limit_gradient!, backend, workgroup, ndrange)
    kernel!(method, K, power, x, y, z, F, cells, cell_neighbours, cell_faces, faces)
    sync!(∇F.result, mesh, config) # ghost limiter values are wrong (partial face lists)
end

function limit_gradient!(method::Venkatakrishnan, ∇F, F::VectorField, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    mesh = F.mesh
    (; cells, cell_neighbours, cell_faces, faces) = mesh
    TF = _get_float(mesh)
    K = TF(method.K)
    power = TF(_cell_length_power(mesh))

    (; xx, xy, xz) = ∇F.result
    (; yx, yy, yz) = ∇F.result
    (; zx, zy, zz) = ∇F.result

    ndrange = length(cells)
    kernel! = _sized(_limit_gradient!, backend, workgroup, ndrange)
    kernel!(method, K, power, xx, xy, xz, F.x, cells, cell_neighbours, cell_faces, faces)
    kernel!(method, K, power, yx, yy, yz, F.y, cells, cell_neighbours, cell_faces, faces)
    kernel!(method, K, power, zx, zy, zz, F.z, cells, cell_neighbours, cell_faces, faces)
    sync!(∇F.result, mesh, config) # ghost limiter values are wrong (partial face lists)
end

@kernel function _limit_gradient!(
    ::Venkatakrishnan, K, power, x, y, z, F, cells, cell_neighbours, cell_faces, faces)
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
        h = cells.volume[cID]^power
        Kh = K*h
        eps2 = Kh*Kh*Kh
        two = one(phiP) + one(phiP)
        limiter = one(phiP) # also clips the smooth function, which exceeds 1 for ratios above 2
        ϵ = 10 * eps(phiP)

        for fi ∈ faces_range
            fID = cell_faces[fi]

            fc = faces.centre[fID]
            δϕ = (fc - cc)⋅grad0

            if abs(δϕ) > ϵ
                Δ = ifelse(δϕ > zero(δϕ), phiMax - phiP, phiMin - phiP)
                ΔΔ = Δ*Δ
                # floatmin: with ε = 0 and φ_P = 0 the guard ϵ underflows and δϕ² can be 0 (0/0)
                den = max(ΔΔ + two*δϕ*δϕ + δϕ*Δ + eps2, floatmin(phiP))
                ψ = (ΔΔ + eps2 + two*δϕ*Δ)/den
                limiter = min(limiter, ψ)
            end
        end
        grad0 *= limiter
        x.values[cID] = grad0[1]
        y.values[cID] = grad0[2]
        z.values[cID] = grad0[3]
    end
end
