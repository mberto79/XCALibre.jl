# Explicit part of the viscous stress in the momentum equation.
#
# With τ = μ_eff (∇U + (∇U)ᵀ - ⅔ (∇·U) I) the momentum equation needs ∇·τ. Its first part,
# ∇·(μ_eff ∇U), is the implicit Laplacian of each solver's U equation; the rest,
# ∇·(μ_eff dev2((∇U)ᵀ)) with dev2(A) = A - ⅔ tr(A) I, is added here as an explicit source.
# It vanishes for constant μ_eff in incompressible flow (∇·(∇U)ᵀ = ∇(∇·U) = 0) and is
# needed once μ_eff varies, e.g. with a turbulence model.

"""
    stress_source(mesh)

Source field of the U equation that holds ∇·(μ_eff dev2((∇U)ᵀ)). Its components keep the mesh
so that `transpose_stress!` writes the divergence into them directly.
"""
stress_source(mesh) = VectorField(ScalarField(mesh), ScalarField(mesh), ScalarField(mesh), mesh)

"""
    stress_fluxes(mesh, transpose_stress::Bool)

Face fluxes (one `FaceScalarField` per component) used by `transpose_stress!`, or `nothing`
when the term is switched off, in which case the source stays zero.
"""
stress_fluxes(mesh, transpose_stress::Bool) =
    transpose_stress ? ntuple(_ -> FaceScalarField(mesh), 3) : nothing

"""
    transpose_stress!(source, fluxes, mueff, gradU, U_BCs, config)

Evaluate `source` = ∇·(μ_eff dev2((∇U)ᵀ)) from the cell gradient `gradU` and the face
viscosity `mueff` (ν_eff for the incompressible solvers). The face flux is
μ_eff,f ((∇U)ᵀ_f n - ⅔ tr(∇U)_f n) A_f with ∇U_f interpolated linearly between the two cells
(owner value on boundary faces); it is zero on slip, symmetry and empty boundaries.
"""
function transpose_stress!(source, fluxes, mueff, gradU, U_BCs, config)
    mesh = source.mesh
    # the ghost cells of a distributed mesh feed the processor faces: exchange ∇U row by row
    t = gradU.result
    sync!(VectorField(t.xx, t.xy, t.xz, mesh), mesh, config)
    sync!(VectorField(t.yx, t.yy, t.yz, mesh), mesh, config)
    sync!(VectorField(t.zx, t.zy, t.zz, mesh), mesh, config)

    fx, fy, fz = fluxes
    explicit_shear_stress!(fx, fy, fz, mueff, gradU, U_BCs, config)
    div!(source.x, fx, config)
    div!(source.y, fy, config)
    div!(source.z, fz, config)
    nothing
end
transpose_stress!(source, ::Nothing, mueff, gradU, U_BCs, config) = nothing

function explicit_shear_stress!(mugradUTx::FaceScalarField, mugradUTy::FaceScalarField, mugradUTz::FaceScalarField, mueff, gradU, U_BCs, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    (; faces, boundary_cellsID) = mugradUTx.mesh

    n_faces = length(faces)
    n_bfaces = length(boundary_cellsID)
    n_ifaces = n_faces - n_bfaces

    ndrange = n_ifaces
    kernel! = _sized(_explicit_shear_stress_internal!, backend, workgroup, ndrange)
    kernel!(mugradUTx, mugradUTy, mugradUTz, mueff, gradU, faces, n_bfaces)
    KernelAbstractions.synchronize(backend)

    ndrange=n_bfaces
    kernel! = _sized(_explicit_shear_stress_boundaries!, backend, workgroup, ndrange)
    kernel!(mugradUTx, mugradUTy, mugradUTz, mueff, gradU, faces)
    KernelAbstractions.synchronize(backend)

    for BC ∈ U_BCs
        zero_explicit_stress!(BC, mugradUTx, mugradUTy, mugradUTz, backend, workgroup)
    end
end

zero_explicit_stress!(BC, mugradUTx, mugradUTy, mugradUTz, backend, workgroup) = nothing

function zero_explicit_stress!(
    BC::Union{Slip,Symmetry,Empty}, mugradUTx, mugradUTy, mugradUTz, backend, workgroup)
    (; IDs_range) = BC
    ndrange = length(IDs_range)
    ndrange == 0 && return nothing
    kernel! = _zero_explicit_stress!(backend)
    kernel!(mugradUTx, mugradUTy, mugradUTz, IDs_range;
        _dynamic_setup(backend, workgroup, ndrange)...)
    KernelAbstractions.synchronize(backend)
end

@kernel function _zero_explicit_stress!(mugradUTx, mugradUTy, mugradUTz, IDs_range)
    i = @index(Global)
    fID = IDs_range[i]
    mugradUTx[fID] = 0
    mugradUTy[fID] = 0
    mugradUTz[fID] = 0
end

@kernel function _explicit_shear_stress_internal!(
    mugradUTx, mugradUTy, mugradUTz, mueff, gradU, faces, n_bfaces)
    i = @index(Global)

    fID = i + n_bfaces
    face = faces[fID]
    (; area, normal, ownerCells) = face
    cID1 = ownerCells[1]
    cID2 = ownerCells[2]
    F = typeof(area)

    # Linear interpolation of gradU at the face
    gradUf = F(0.5)*(gradU[cID1] + gradU[cID2])

    # Explicit part of the stress projection: mu * ( (grad U)^T . n - 2/3 * (div U) * n )
    divU = sum(diag(gradUf))
    projection = transpose(gradUf)*normal - (F(2)/3*divU)*normal

    mueffi = mueff[fID]
    mugradUTx[fID] = mueffi*projection[1]*area
    mugradUTy[fID] = mueffi*projection[2]*area
    mugradUTz[fID] = mueffi*projection[3]*area
end

@kernel function _explicit_shear_stress_boundaries!(
    mugradUTx, mugradUTy, mugradUTz, mueff, gradU, faces)
    fID = @index(Global)

    face = faces[fID]
    (; area, normal, ownerCells) = face
    cID1 = ownerCells[1]
    gradUi = gradU[cID1]
    F = typeof(area)

    # Explicit part of the stress projection at boundary: mu * ( (grad U)^T . n - 2/3 * (div U) * n )
    divUi = sum(diag(gradUi))
    projection = transpose(gradUi)*normal - (F(2)/3*divUi)*normal

    mueffi = mueff[fID]
    mugradUTx[fID] = mueffi*projection[1]*area
    mugradUTy[fID] = mueffi*projection[2]*area
    mugradUTz[fID] = mueffi*projection[3]*area
end
