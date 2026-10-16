export least_squares!

# Cell gradient from min Σ_f w_f (φ_f - φ_P - ∇φ·d_f)², d_f from the cell centre to the
# neighbour centre (internal) or face centre (boundary, face value). The inverse of G = Σ w d dᵀ
# is built once per Grad and kept in Grad.correctors; vector fields give ∇ψ[a,b] = ∂ψ_a/∂x_b.

struct LSQGeometry{T<:Tuple} # G⁻¹ per cell: xx, yy, zz, xy, xz, yz
    Ginv::T
end
Adapt.@adapt_structure LSQGeometry

@inline _lsq_weight(d2::T) where T = d2 > zero(T) ? inv(d2) : zero(T)

@inline _lsq_sym(a, b, c, d, e, f) = SMatrix{3,3}(a,d,e, d,b,f, e,f,c)

# displacement and weight of face `fID` seen from cell `i`; internal faces go to the neighbour
@inline function _lsq_internal(cells, cell_neighbours, fi, xP)
    d = cells.centre[cell_neighbours[fi]] - xP
    return d, _lsq_weight(d ⋅ d)
end

@inline function _lsq_boundary(cells, faces, fID, cID)
    d = faces.centre[fID] - cells.centre[cID]
    return d, _lsq_weight(d ⋅ d)
end

# GEOMETRY ---------------------------------------------------------------------------------

function _lsq_geometry(mesh)
    TF = eltype(mesh.cells.volume)
    backend = get_backend(mesh.cells.volume)
    n = length(mesh.cells)
    nb = length(mesh.boundary_cells)
    G = ntuple(_ -> KernelAbstractions.zeros(backend, TF, n), 6)
    kernel! = _sized(_lsq_geometry_internal!, backend, 256, n)
    kernel!(G..., mesh.cells, mesh.cell_faces, mesh.cell_neighbours)
    kernel! = _sized(_lsq_geometry_boundary!, backend, 256, nb)
    kernel!(G..., mesh.cells, mesh.faces, mesh.boundary_cells, mesh.boundary_cell_faces_range, mesh.boundary_cell_faces)
    kernel! = _sized(_lsq_geometry_invert!, backend, 256, n)
    kernel!(G...)
    return LSQGeometry(G)
end

@kernel inbounds=true function _lsq_geometry_internal!(
    gxx, gyy, gzz, gxy, gxz, gyz, cells, cell_faces, cell_neighbours)
    i = @index(Global)
    xP = cells.centre[i]
    TF = typeof(cells.volume[i])
    z = zero(TF)
    G = SMatrix{3,3,TF,9}(z,z,z,z,z,z,z,z,z)
    for fi ∈ cells.faces_range[i]
        d, w = _lsq_internal(cells, cell_neighbours, fi, xP)
        G += w*(d*d')
    end
    gxx[i] = G[1,1]; gyy[i] = G[2,2]; gzz[i] = G[3,3]
    gxy[i] = G[1,2]; gxz[i] = G[1,3]; gyz[i] = G[2,3]
end

# boundary kernels: one work item per boundary cell adds its faces in face order (deterministic)
@kernel inbounds=true function _lsq_geometry_boundary!(
    gxx, gyy, gzz, gxy, gxz, gyz, cells, faces, boundary_cells, boundary_cell_faces_range, boundary_cell_faces)
    j = @index(Global)
    cID = boundary_cells[j]
    for k ∈ boundary_cell_faces_range[j]
        fID = boundary_cell_faces[k]
        d, w = _lsq_boundary(cells, faces, fID, cID)
        G = w*(d*d')
        gxx[cID] += G[1,1]
        gyy[cID] += G[2,2]
        gzz[cID] += G[3,3]
        gxy[cID] += G[1,2]
        gxz[cID] += G[1,3]
        gyz[cID] += G[2,3]
    end
end

@kernel inbounds=true function _lsq_geometry_invert!(gxx, gyy, gzz, gxy, gxz, gyz)
    i = @index(Global)
    TF = typeof(gxx[i])
    zr = zero(TF); o = one(TF)
    G = _lsq_sym(gxx[i], gyy[i], gzz[i], gxy[i], gxz[i], gyz[i])
    # Remove directions without data (2D meshes) so G stays invertible. Slivers (nearly collinear
    # neighbours) give a noisy gradient along the poorly sampled direction: not regularised, as a shift
    # of G would also bias the exactness for linear fields.
    tol = TF(100)*eps(TF)*(G[1,1] + G[2,2] + G[3,3])
    G += SMatrix{3,3,TF,9}(
        G[1,1] <= tol ? o : zr, zr, zr,
        zr, G[2,2] <= tol ? o : zr, zr,
        zr, zr, G[3,3] <= tol ? o : zr)
    Gi = inv(G)
    gxx[i] = Gi[1,1]; gyy[i] = Gi[2,2]; gzz[i] = Gi[3,3]
    gxy[i] = Gi[1,2]; gxz[i] = Gi[1,3]; gyz[i] = Gi[2,3]
end

# SCALAR FIELD -> VECTOR GRADIENT ------------------------------------------------------------

function least_squares!(grad::Grad{LeastSquares,F,R,I,M}, phif, phi, config) where {F,R<:VectorField,I<:LSQGeometry,M}
    (; backend, workgroup) = config.hardware
    (; x, y, z) = grad.result
    (; mesh) = phi
    n = length(x)
    nbcells = length(mesh.boundary_cells)

    kernel! = _sized(_lsq_internal_scalar!, backend, workgroup, n)
    kernel!(x, y, z, phi, mesh.cells, mesh.cell_faces, mesh.cell_neighbours)

    kernel! = _sized(_lsq_boundary_scalar!, backend, workgroup, nbcells)
    kernel!(x, y, z, phif, phi, mesh.cells, mesh.faces, mesh.boundary_cells, mesh.boundary_cell_faces_range, mesh.boundary_cell_faces)

    kernel! = _sized(_lsq_solve_scalar!, backend, workgroup, n)
    kernel!(x, y, z, grad.correctors.Ginv...)
    sync!(grad.result, mesh, config) # self-syncing seam (no-op serial)
    return nothing
end

@kernel inbounds=true function _lsq_internal_scalar!(
    dx, dy, dz, phi, cells, cell_faces, cell_neighbours)
    i = @index(Global)
    xP = cells.centre[i]
    phiP = phi[i]
    TF = typeof(cells.volume[i])
    b = zero(SVector{3,TF})
    for fi ∈ cells.faces_range[i]
        nID = cell_neighbours[fi]
        d, w = _lsq_internal(cells, cell_neighbours, fi, xP)
        b += w*(phi[nID] - phiP)*d
    end
    dx.values[i] = b[1]; dy.values[i] = b[2]; dz.values[i] = b[3]
end

@kernel inbounds=true function _lsq_boundary_scalar!(dx, dy, dz, phif, phi, cells, faces, boundary_cells, boundary_cell_faces_range, boundary_cell_faces)
    j = @index(Global)
    cID = boundary_cells[j]
    for k ∈ boundary_cell_faces_range[j]
        fID = boundary_cell_faces[k]
        d, w = _lsq_boundary(cells, faces, fID, cID)
        b = w*(phif[fID] - phi[cID])*d
        dx.values[cID] += b[1]
        dy.values[cID] += b[2]
        dz.values[cID] += b[3]
    end
end

@kernel inbounds=true function _lsq_solve_scalar!(dx, dy, dz, gxx, gyy, gzz, gxy, gxz, gyz)
    i = @index(Global)
    Gi = _lsq_sym(gxx[i], gyy[i], gzz[i], gxy[i], gxz[i], gyz[i])
    g = Gi*SVector(dx.values[i], dy.values[i], dz.values[i])
    dx.values[i] = g[1]; dy.values[i] = g[2]; dz.values[i] = g[3]
end

# VECTOR FIELD -> TENSOR GRADIENT ------------------------------------------------------------

function least_squares!(grad::Grad{LeastSquares,F,R,I,M}, psif, psi, config) where {F,R<:TensorField,I<:LSQGeometry,M}
    (; backend, workgroup) = config.hardware
    (; xx, xy, xz, yx, yy, yz, zx, zy, zz) = grad.result
    (; mesh) = psi
    n = length(xx)
    nbcells = length(mesh.boundary_cells)

    kernel! = _sized(_lsq_internal_vector!, backend, workgroup, n)
    kernel!(xx, xy, xz, yx, yy, yz, zx, zy, zz, psi, mesh.cells, mesh.cell_faces, mesh.cell_neighbours)

    kernel! = _sized(_lsq_boundary_vector!, backend, workgroup, nbcells)
    kernel!(xx, xy, xz, yx, yy, yz, zx, zy, zz, psif, psi, mesh.cells, mesh.faces, mesh.boundary_cells, mesh.boundary_cell_faces_range, mesh.boundary_cell_faces)

    kernel! = _sized(_lsq_solve_vector!, backend, workgroup, n)
    kernel!(xx, xy, xz, yx, yy, yz, zx, zy, zz, grad.correctors.Ginv...)
    sync!(grad.result, mesh, config)
    return nothing
end

@kernel inbounds=true function _lsq_internal_vector!(
    xx, xy, xz, yx, yy, yz, zx, zy, zz, psi, cells, cell_faces, cell_neighbours)
    i = @index(Global)
    xP = cells.centre[i]
    psiP = psi[i]
    TF = typeof(cells.volume[i])
    z = zero(TF)
    b = SMatrix{3,3,TF,9}(z,z,z,z,z,z,z,z,z)
    for fi ∈ cells.faces_range[i]
        nID = cell_neighbours[fi]
        d, w = _lsq_internal(cells, cell_neighbours, fi, xP)
        b += w*((psi[nID] - psiP)*d')
    end
    xx.values[i] = b[1,1]; xy.values[i] = b[1,2]; xz.values[i] = b[1,3]
    yx.values[i] = b[2,1]; yy.values[i] = b[2,2]; yz.values[i] = b[2,3]
    zx.values[i] = b[3,1]; zy.values[i] = b[3,2]; zz.values[i] = b[3,3]
end

@kernel inbounds=true function _lsq_boundary_vector!(
    xx, xy, xz, yx, yy, yz, zx, zy, zz, psif, psi, cells, faces, boundary_cells, boundary_cell_faces_range, boundary_cell_faces)
    j = @index(Global)
    cID = boundary_cells[j]
    for k ∈ boundary_cell_faces_range[j]
        fID = boundary_cell_faces[k]
        d, w = _lsq_boundary(cells, faces, fID, cID)
        b = w*((psif[fID] - psi[cID])*d')
        xx.values[cID] += b[1,1]
        xy.values[cID] += b[1,2]
        xz.values[cID] += b[1,3]
        yx.values[cID] += b[2,1]
        yy.values[cID] += b[2,2]
        yz.values[cID] += b[2,3]
        zx.values[cID] += b[3,1]
        zy.values[cID] += b[3,2]
        zz.values[cID] += b[3,3]
    end
end

@kernel inbounds=true function _lsq_solve_vector!(
    xx, xy, xz, yx, yy, yz, zx, zy, zz, gxx, gyy, gzz, gxy, gxz, gyz)
    i = @index(Global)
    Gi = _lsq_sym(gxx[i], gyy[i], gzz[i], gxy[i], gxz[i], gyz[i])
    b = SMatrix{3,3}(
        xx.values[i], yx.values[i], zx.values[i],
        xy.values[i], yy.values[i], zy.values[i],
        xz.values[i], yz.values[i], zz.values[i])
    g = b*Gi
    xx.values[i] = g[1,1]; xy.values[i] = g[1,2]; xz.values[i] = g[1,3]
    yx.values[i] = g[2,1]; yy.values[i] = g[2,2]; yz.values[i] = g[2,3]
    zx.values[i] = g[3,1]; zy.values[i] = g[3,2]; zz.values[i] = g[3,3]
end

# Empty patches (2D cases on a one-cell-thick 3D mesh) store zero in the face field and carry
# no information: take the owner-cell value there, so the faces add no z-variation to the fit.
function _lsq_empty_faces!(phif, phi, BCs, config)
    (; backend, workgroup) = config.hardware
    for BC ∈ BCs
        BC isa Empty || continue
        range = BC.IDs_range
        kernel! = _sized(_lsq_empty_faces_kernel!, backend, workgroup, length(range))
        kernel!(phif, phi, first(range) - 1)
    end
    return nothing
end

@kernel inbounds=true function _lsq_empty_faces_kernel!(phif, phi, offset)
    i = @index(Global)
    @uniform boundary_cellsID = phi.mesh.boundary_cellsID
    fID = i + offset
    phif[fID] = phi[boundary_cellsID[fID]]
end
