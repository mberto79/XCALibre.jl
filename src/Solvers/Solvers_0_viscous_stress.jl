# Explicit part of the viscous stress in the momentum equation.
#
# With τ = μ_eff (∇U + (∇U)ᵀ - ⅔ (∇·U) I) the momentum equation needs ∇·τ. Its first part,
# ∇·(μ_eff ∇U), is the implicit Laplacian of each solver's U equation; the rest,
# ∇·(μ_eff dev2((∇U)ᵀ)) with dev2(A) = A - ⅔ tr(A) I, is added here as an explicit source.
# It vanishes for constant μ_eff in incompressible flow (∇·(∇U)ᵀ = ∇(∇·U) = 0) and is
# needed once μ_eff varies, e.g. with a turbulence model.

"""
    transpose_stress!(source, mueff, gradU, U_BCs, config)

Set `source` = ∇·(μ_eff dev2((∇U)ᵀ)) from the cell gradient `gradU` and the face viscosity
`mueff` (ν_eff for the incompressible solvers), using the current values of both.
"""
function transpose_stress!(source, mueff, gradU, U_BCs, config)
    mesh = source.mesh
    # ghost cells of a distributed mesh feed the processor faces: exchange ∇U row by row
    t = gradU.result
    sync!(VectorField(t.xx, t.xy, t.xz, mesh), mesh, config)
    sync!(VectorField(t.yx, t.yy, t.yz, mesh), mesh, config)
    sync!(VectorField(t.zx, t.zy, t.zz, mesh), mesh, config)
    div!(source, mueff, Dev2(T(gradU)), U_BCs, config)
end
