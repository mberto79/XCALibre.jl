# Explicit part of the viscous stress in the momentum equation.
#
# With τ = μ_eff (∇U + (∇U)ᵀ - ⅔ (∇·U) I) the momentum equation needs ∇·τ. Its first part,
# ∇·(μ_eff ∇U), is the implicit Laplacian of each solver's U equation; the rest,
# ∇·(μ_eff dev2((∇U)ᵀ)) with dev2(A) = A - ⅔ tr(A) I, is added here as an explicit source.
# It vanishes for constant μ_eff in incompressible flow (∇·(∇U)ᵀ = ∇(∇·U) = 0) and is
# needed once μ_eff varies, e.g. with a turbulence model.

"""
    transpose_stress!(source, mueff, gradU, U_BCs, config; cell_mueff=nothing)

Set `source` = ∇·(μ_eff dev2((∇U)ᵀ)) from the cell gradient `gradU` and the face viscosity
`mueff` (ν_eff for the incompressible solvers), using the current values of both. With the
cell viscosity `cell_mueff` (see `cell_nueff` and `cell_mueff`) internal faces interpolate the
product μ_eff dev2((∇U)ᵀ) of the two cells; `mueff` is then used on boundary faces only.
"""
function transpose_stress!(source, mueff, gradU, U_BCs, config; cell_mueff=nothing)
    mesh = source.mesh
    # ghost cells of a distributed mesh feed the processor faces: exchange ∇U row by row
    t = gradU.result
    sync!(VectorField(t.xx, t.xy, t.xz, mesh), mesh, config)
    sync!(VectorField(t.yx, t.yy, t.yz, mesh), mesh, config)
    sync!(VectorField(t.zx, t.zy, t.zz, mesh), mesh, config)
    _sync_cell_viscosity!(cell_mueff, mesh, config)
    div!(source, mueff, Dev2(T(gradU)), U_BCs, config; Γc=cell_mueff)
end

# Cell μ_eff = ρ(ν + ν_t), evaluated on access. `rho === nothing` gives the kinematic ν_eff of
# the incompressible solvers and `nut === nothing` laminar flow; `nu` and `rho` may be constant
# (ConstantScalar) or cell fields.
struct CellViscosity{R,N,F}
    rho::R
    nu::N
    nut::F
end
Adapt.@adapt_structure CellViscosity

@inline _add_nut(::Nothing, i, nu) = nu
@inline _add_nut(nut, i, nu) = nu + nut[i]
@inline _times_rho(::Nothing, i, nueff) = nueff
@inline _times_rho(rho, i, nueff) = rho[i]*nueff
Base.getindex(c::CellViscosity, i::Integer) = _times_rho(c.rho, i, _add_nut(c.nut, i, c.nu[i]))

_cell_nut(turbulence) = hasproperty(turbulence, :nut) ? turbulence.nut : nothing

_sync_cell!(field::ScalarField, mesh, config) = sync!(field, mesh, config)
_sync_cell!(field, mesh, config) = nothing
_sync_cell_viscosity!(c::CellViscosity, mesh, config) = begin
    _sync_cell!(c.rho, mesh, config); _sync_cell!(c.nu, mesh, config); _sync_cell!(c.nut, mesh, config)
end
_sync_cell_viscosity!(c, mesh, config) = nothing

"""
    cell_nueff(nu, turbulence)

Cell ν_eff = ν + ν_t for `transpose_stress!` in the incompressible solvers (ν alone for
laminar flow). `nu` must be a cell value (constant or cell field); `nothing` otherwise, and
the term then uses the face viscosity at every face.
"""
cell_nueff(nu::Union{ConstantScalar,ScalarField}, turbulence) =
    CellViscosity(nothing, nu, _cell_nut(turbulence))
cell_nueff(nu, turbulence) = nothing

"""
    cell_mueff(rho, nu, turbulence)

Cell μ_eff = ρ(ν + ν_t) for `transpose_stress!` in the compressible and multiphase solvers,
the cell counterpart of their face `mueff = rhof*(nuf + nutf)`. `rho` and `nu` must be cell
values (constant or cell fields); `nothing` otherwise.
"""
cell_mueff(rho::Union{ConstantScalar,ScalarField}, nu::Union{ConstantScalar,ScalarField}, turbulence) =
    CellViscosity(rho, nu, _cell_nut(turbulence))
cell_mueff(rho, nu, turbulence) = nothing

"""
    prime_face_nut!(model, boundaries, config)

Set the face eddy viscosity from the current cell values at the start of every run, as
`turbulence!` does: interpolated to the faces, boundary values from the conditions, and the
wall-function values on wall faces. `turbulence!` updates the face values only after the first
momentum solve, so without this the first iterations would see face values that do not match
the cell ν_t used by the explicit transpose stress: zero after initialisation (the implicit
Laplacian then holds ν alone while the explicit term carries the initial ν_t, typically
10³-10⁴ ν), or values left by an earlier run on fields since re-initialised. Either makes the
explicit part dominate the first iterations, which can trigger a divergence.
"""
function prime_face_nut!(model, boundaries, config)
    turbulence = model.turbulence
    hasproperty(turbulence, :nut) && hasproperty(turbulence, :nutf) &&
        hasproperty(boundaries, :nut) || return nothing
    (; nut, nutf) = turbulence
    interpolate!(nutf, nut, config)
    correct_boundaries!(nutf, nut, boundaries.nut, zero(_get_float(nut.mesh)), config)
    # wall faces take their wall-function value (zero below yPlusLam), not the cell value
    ModelPhysics.correct_eddy_viscosity!(nutf, boundaries.nut, model, config)
    nothing
end
