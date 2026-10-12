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
product μ_eff dev2((∇U)ᵀ) of the two cells, and boundary faces use `mueff` times the
boundary-face gradient (see `_transpose_stress_boundary!`).
"""
function transpose_stress!(source, mueff, gradU, U_BCs, config; cell_mueff=nothing)
    mesh = source.mesh
    # ghost cells of a distributed mesh feed the processor faces: exchange ∇U row by row
    t = gradU.result
    sync!(VectorField(t.xx, t.xy, t.xz, mesh), mesh, config)
    sync!(VectorField(t.yx, t.yy, t.yz, mesh), mesh, config)
    sync!(VectorField(t.zx, t.zy, t.zz, mesh), mesh, config)
    _sync_cell_viscosity!(cell_mueff, mesh, config)
    tensor = Dev2(transpose_values(t))
    isnothing(cell_mueff) && return div!(source, mueff, tensor, U_BCs, config)
    div!(source, mueff, tensor, (), config; Γc=_kernel_values(cell_mueff)) # internal faces
    (; backend, workgroup) = config.hardware
    sourcev, mueffv, gradUv, Uv = field_values(source), _kernel_values(mueff), field_values(t), field_values(gradU.field)
    for BC ∈ U_BCs
        _transpose_stress_boundary!(sourcev, mueffv, gradUv, Uv, BC, mesh, backend, workgroup)
    end
    nothing
end

# Boundary faces: owner-cell gradient with its normal derivative replaced by the boundary
# condition's: (U_b - U_c)/delta (fixed value), 0 (zero gradient), -(U_c·n)n/delta (slip,
# symmetry). Empty faces carry no flux; other conditions keep the owner-cell gradient.
_transpose_stress_boundary!(source, mueff, gradU, U, ::Empty, mesh, backend, workgroup) = nothing

function _transpose_stress_boundary!(source, mueff, gradU, U, BC, mesh, backend, workgroup)
    (; IDs_range) = BC
    isempty(IDs_range) && return nothing
    (; cells, faces) = mesh
    kernel! = _sized(_transpose_stress_boundary_kernel!, backend, workgroup, length(IDs_range))
    kernel!(source, mueff, gradU, U, BC, IDs_range, cells, faces)
    KernelAbstractions.synchronize(backend)
end

# face-normal gradient of U, or nothing to keep the owner-cell normal derivative
@inline _boundary_sngrad(BC::Union{Wall,Dirichlet}, Uc, delta) =
    (SVector{3}(BC.value[1], BC.value[2], BC.value[3]) - Uc)/delta
@inline _boundary_sngrad(::Union{Zerogradient,Extrapolated}, Uc, delta) = zero(Uc)
# mirror condition: face value U_c - (U_c·n)n, normal gradient -(U_c·n)n/delta
@inline _boundary_sngrad(::Union{Slip,Symmetry}, Uc, delta, normal) = -(Uc⋅normal)*normal/delta
@inline _boundary_sngrad(BC, Uc, delta, normal) = _boundary_sngrad(BC, Uc, delta)
@inline _boundary_sngrad(BC, Uc, delta) = nothing

@inline _patch_gradient(Mc, ::Nothing, normal) = Mc
@inline _patch_gradient(Mc, sngrad, normal) = Mc + (sngrad - Mc*normal)*normal'

@kernel inbounds=true function _transpose_stress_boundary_kernel!(
    source, mueff, gradU, U, BC, IDs_range, cells, faces)
    i = @index(Global)
    fID = IDs_range[i]
    cID = faces.ownerCells[fID][1]
    normal, area, delta = faces.normal[fID], faces.area[fID], faces.delta[fID]
    TF = typeof(area)
    Mc = gradU[cID] # Mc[i,j] = ∂U_i/∂x_j
    Mb = _patch_gradient(Mc, _boundary_sngrad(BC, U[cID], delta, normal), normal)
    Tb = Mb' - TF(2)/3*tr(Mb)*I
    flux = mueff[fID]*(Tb*normal)*(area/cells.volume[cID])
    Atomix.@atomic source.x[cID] += flux[1]
    Atomix.@atomic source.y[cID] += flux[2]
    Atomix.@atomic source.z[cID] += flux[3]
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
@inline Base.getindex(c::CellViscosity, i::Integer) = _times_rho(c.rho, i, _add_nut(c.nut, i, c.nu[i]))

_cell_nut(turbulence) = hasproperty(turbulence, :nut) ? turbulence.nut : nothing

_sync_cell!(field::ScalarField, mesh, config) = sync!(field, mesh, config)
_sync_cell!(field, mesh, config) = nothing
_sync_cell_viscosity!(c::CellViscosity, mesh, config) = begin
    _sync_cell!(c.rho, mesh, config); _sync_cell!(c.nu, mesh, config); _sync_cell!(c.nut, mesh, config)
end
_sync_cell_viscosity!(c, mesh, config) = nothing
_kernel_values(c::CellViscosity) = CellViscosity(_kernel_values(c.rho), _kernel_values(c.nu), _kernel_values(c.nut))

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
