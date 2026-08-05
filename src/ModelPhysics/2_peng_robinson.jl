export PengRobinson, PengRobinsonBeta
export pr_compressibility_factor, pr_density, pr_branch_exists

# =============================================================================
#  Peng-Robinson cubic equation of state
# =============================================================================
#
#      p = R*T/(v - b)  -  a*alpha(T)/(v^2 + 2*b*v - b^2)
#
#  WHY THIS EXISTS ALONGSIDE THE HELMHOLTZ EOS
#
#  The Helmholtz EOS is more accurate, but it cannot be evaluated in a kernel: it
#  Newton-solves for density and allocates ~20 vectors per call. So it has to be
#  tabulated, and tabulation is where the trouble is - a rectangular (p,T) grid
#  necessarily includes states where the requested branch does not exist, and the
#  fallbacks used there have produced discontinuous and non-monotonic density
#  tables (see `validate_property_table`).
#
#  A cubic has none of that. It is ANALYTIC: Cardano gives the roots in closed
#  form, with no iteration, no allocation and no table. It runs directly in a
#  kernel at every cell, every step, so `rho(p,T)` is exactly the same smooth
#  function everywhere in the domain and there is no interpolation error and no
#  off-branch patching to get wrong.
#
#  That makes it the right tool for TESTING A SOLVER against a variable-density
#  equation of state: any misbehaviour is then the solver's, not the property
#  data's.
#
#  ACCURACY - READ THIS BEFORE USING IT FOR PHYSICS
#
#  Peng-Robinson is a generic cubic fitted to hydrocarbons. For hydrogen it is
#  poor: H2 is a quantum fluid with a NEGATIVE acentric factor (omega = -0.219),
#  which is outside the range the alpha-function correlation was fitted over.
#  Expect liquid densities wrong by tens of percent. It is here to give the
#  solver a smooth, cheap, analytic rho(p,T) - NOT to replace the Helmholtz EOS
#  for quantitative work.
#
#  WHAT IT DOES AND DOES NOT SUPPLY
#
#  `rho`, `psi` and (via `PengRobinsonBeta`) `beta` come from the cubic. `cp`,
#  `k` and `mu` do NOT: a cubic says nothing about transport properties, and its
#  cp needs an ideal-gas correlation it does not carry. Pair it with `ConstCp`,
#  `ConstK` and `ConstMu`, or with tabulated versions of those.
# =============================================================================

"""
    PengRobinson <: AbstractEosModel

Peng-Robinson cubic equation of state for one branch (liquid or vapour).

Construct from critical properties,

    PengRobinson(Tc=33.145, pc=1.2964e6, omega=-0.219, M=2.01588e-3, branch=:vapour)

or from a fluid this package already knows,

    PengRobinson(H2(), branch=:liquid)

### Keywords
- `Tc`, `pc` -- critical temperature [K] and pressure [Pa].
- `omega`    -- acentric factor [-].
- `M` or `R` -- molar mass [kg/mol] or specific gas constant [J/kg/K].
- `branch`   -- `:liquid` (smallest root) or `:vapour` (largest root).

### Branch selection and where a branch stops existing

Below the critical temperature the cubic has three real roots: the smallest is
the liquid, the largest the vapour, and the middle one is thermodynamically
unstable and never returned.

Above `Tc`, or beyond the spinodal where the metastable extension of a branch
ends, there is only ONE real root and both branches return it. That is correct
thermodynamics - there is no liquid above `Tc` - but it means `rho` still has a
step where a branch ceases to exist, exactly as the tabulated path does. The
difference is that here it is analytic and locatable: [`pr_branch_exists`](@ref)
reports it, so a case can be set up to stay inside the region where the branch
is real rather than discovering the step at run time.

### Fields
Stored pre-combined so the kernel does no setup work: `a` and `b` are the
mass-specific attraction and covolume parameters, `kappa` the alpha-function
coefficient.
"""
struct PengRobinson{B,F<:AbstractFloat} <: AbstractEosModel
    R::F        # specific gas constant [J/kg/K]
    Tc::F       # critical temperature [K]
    a::F        # attraction parameter [Pa*(m^3/kg)^2]
    b::F        # covolume [m^3/kg]
    kappa::F    # alpha-function coefficient [-]
end

# Acentric factors for the fluids with a Helmholtz model in this package.
# Hydrogen's is negative, which is precisely why Peng-Robinson struggles with it.
_acentric_factor(::H2)      = -0.219
_acentric_factor(::H2_para) = -0.219
_acentric_factor(::N2)      =  0.0372

function PengRobinson(; Tc, pc, omega, M=nothing, R=nothing, branch::Symbol=:vapour)
    branch in (:liquid, :vapour) || throw(ArgumentError(
        "`branch` must be :liquid or :vapour, got :$branch"))
    if (M === nothing) == (R === nothing)
        throw(ArgumentError("`PengRobinson` needs exactly one of `M=` [kg/mol] or `R=` [J/kg/K]"))
    end
    Rs = R === nothing ? R_UNIVERSAL/M : float(R)
    Tc, pc, omega = float(Tc), float(pc), float(omega)

    # Standard Peng-Robinson (1976) coefficients. Written in MASS-specific units:
    # the form is identical to the molar one provided R, a and b use the same
    # basis throughout.
    a = 0.45724*Rs^2*Tc^2/pc
    b = 0.07780*Rs*Tc/pc
    kappa = 0.37464 + 1.54226*omega - 0.26992*omega^2

    F = typeof(Rs)
    return PengRobinson{branch,F}(Rs, F(Tc), F(a), F(b), F(kappa))
end

PengRobinson(fluid::HelmholtzEnergyFluid; branch::Symbol=:vapour) = begin
    c = helmholtz_constants(fluid, Float64)
    PengRobinson(Tc=c.T_c, pc=c.p_c, omega=_acentric_factor(fluid), M=c.M, branch=branch)
end

_pr_is_liquid(::PengRobinson{:liquid}) = true
_pr_is_liquid(::PengRobinson{:vapour}) = false

"""
    pr_compressibility_factor(eos, p, T) -> (Z, n_real_roots)

Compressibility factor `Z = p*v/(R*T)` for the model's branch, and how many real
roots the cubic had (1 or 3).

Solved by Cardano's formula - closed form, no iteration, no allocation - which is
what makes this callable from a kernel. `n_real_roots == 1` means the requested
branch does not exist at this state and the single available root was returned.
"""
@inline function pr_compressibility_factor(eos::PengRobinson, p, T)
    (; R, Tc, a, b, kappa) = eos
    TF = typeof(float(p))

    sqrt_Tr = sqrt(T/Tc)
    alpha = (one(TF) + kappa*(one(TF) - sqrt_Tr))^2

    A = a*alpha*p/(R*T)^2
    B = b*p/(R*T)

    # Z^3 + c2*Z^2 + c1*Z + c0 = 0
    c2 = -(one(TF) - B)
    c1 = A - 2*B - 3*B^2
    c0 = -(A*B - B^2 - B^3)

    Q = (3*c1 - c2^2)/9
    Rr = (9*c2*c1 - 27*c0 - 2*c2^3)/54
    D = Q^3 + Rr^2

    if D < zero(TF)
        # Three distinct real roots: trigonometric form.
        theta = acos(clamp(Rr/sqrt(-Q^3), -one(TF), one(TF)))
        m = 2*sqrt(-Q)
        z1 = m*cos(theta/3) - c2/3
        z2 = m*cos((theta + 2*TF(pi))/3) - c2/3
        z3 = m*cos((theta + 4*TF(pi))/3) - c2/3
        zmin = min(z1, min(z2, z3))
        zmax = max(z1, max(z2, z3))
        # A root at or below the covolume B is unphysical (negative free volume);
        # fall back to the vapour root rather than return it.
        Z = _pr_is_liquid(eos) ? (zmin > B ? zmin : zmax) : zmax
        return (Z, 3)
    else
        sqrtD = sqrt(D)
        S = _pr_cbrt(Rr + sqrtD)
        U = _pr_cbrt(Rr - sqrtD)
        return (S + U - c2/3, 1)
    end
end

# `cbrt` of a possibly-negative argument, branch-free and kernel-safe.
@inline _pr_cbrt(x) = sign(x)*abs(x)^(1/3)

"""
    pr_density(eos, p, T) -> kg/m^3

Density of the model's branch, `rho = p/(Z*R*T)`.
"""
@inline function pr_density(eos::PengRobinson, p, T)
    Z, _ = pr_compressibility_factor(eos, p, T)
    return p/(Z*eos.R*T)
end

"""
    pr_branch_exists(eos, p, T) -> Bool

Whether the model's branch is a genuine distinct root at `(p, T)`.

`false` means the cubic has collapsed to a single root - above `Tc`, or beyond
the spinodal - so liquid and vapour are indistinguishable there and `pr_density`
returns the same value for both. Use it to check a case's operating envelope at
setup, where the answer is actionable, rather than meeting the step mid-run.
"""
@inline function pr_branch_exists(eos::PengRobinson, p, T)
    _, n = pr_compressibility_factor(eos, p, T)
    return n == 3
end

# Relative step for the finite-difference derivatives below.
#
# The analytic derivatives of a cubic root are long and easy to get subtly wrong;
# a central difference costs two extra Cardano solves (each a handful of flops,
# no allocation) and cannot disagree with the density the solver actually uses,
# because it is built from the same function. 1e-5 sits well above the noise
# floor of a Float64 closed-form root and well below any curvature scale here.
const _PR_FD = 1e-5

"""
Isothermal compressibility `(1/rho)*(d rho/dp)_T`, by central difference on
[`pr_density`](@ref).
"""
@inline function phase_compressibility(eos::PengRobinson, p_abs, T)
    dp = _PR_FD*p_abs
    rho_p = pr_density(eos, p_abs + dp, T)
    rho_m = pr_density(eos, p_abs - dp, T)
    rho = pr_density(eos, p_abs, T)
    return (rho_p - rho_m)/(2*dp*rho)
end

"""
Thermal expansivity `beta = -(1/rho)*(d rho/dT)_p`, by central difference.
"""
@inline function pr_expansivity(eos::PengRobinson, p_abs, T)
    dT = _PR_FD*T
    rho_p = pr_density(eos, p_abs, T + dT)
    rho_m = pr_density(eos, p_abs, T - dT)
    rho = pr_density(eos, p_abs, T)
    return -(rho_p - rho_m)/(2*dT*rho)
end

# `beta` is carried in a per-cell field filled by `PengRobinsonBeta`, exactly as
# the tabulated path does, so the pressure-work term reads it the same way.
phase_betaT(::PengRobinson, beta, T) = beta*T

specific_gas_constant(eos::PengRobinson) = eos.R

function update_phase_property!(field, model::PengRobinson, p_abs, T, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    ndrange = length(field)
    kernel! = _pr_density!(_setup(backend, workgroup, ndrange)...)
    kernel!(field, p_abs, T, model)
    return nothing
end

@kernel inbounds=true function _pr_density!(rho, p_abs, T, eos)
    i = @index(Global)
    rho[i] = pr_density(eos, p_abs[i], T[i])
end

"""
    PengRobinsonBeta(eos::PengRobinson) <: AbstractExpansivityModel

Thermal expansivity from the same Peng-Robinson model that supplies the density.

Kept as a separate model because `Phase` takes `beta` separately, and sharing the
`eos` object guarantees the two cannot drift apart - the expansivity is a
derivative of exactly the density field being solved with.

    pr  = PengRobinson(H2(), branch=:liquid)
    lh2 = Phase(rho=pr, beta=PengRobinsonBeta(pr), mu=..., k=..., cp=...)
"""
struct PengRobinsonBeta{E} <: AbstractExpansivityModel
    eos::E
end

function update_phase_property!(field, model::PengRobinsonBeta, p_abs, T, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    ndrange = length(field)
    kernel! = _pr_beta!(_setup(backend, workgroup, ndrange)...)
    kernel!(field, p_abs, T, model.eos)
    return nothing
end

@kernel inbounds=true function _pr_beta!(beta, p_abs, T, eos)
    i = @index(Global)
    beta[i] = pr_expansivity(eos, p_abs[i], T[i])
end

"""
    pr_table_report(eos; p, T, np=21, nT=41)

Check a Peng-Robinson branch over an operating envelope before running, and
report where it stops being a distinct root or stops being monotonic in `T`.

The analogue of [`validate_property_table`](@ref) for the analytic path. It
CANNOT fail the way a table does - there is no interpolation and no off-branch
patching - but a branch still ends at the spinodal, and it is far cheaper to
learn that here than from a diverging run.
"""
function pr_table_report(eos::PengRobinson; p, T, np::Integer=21, nT::Integer=41)
    ps = range(float(p[1]), float(p[2]), length=np)
    Ts = range(float(T[1]), float(T[2]), length=nT)

    n_missing = 0
    worst_ratio = 1.0
    worst_at = (0.0, 0.0)
    rho_min, rho_max = Inf, -Inf

    for pp in ps
        prev = NaN
        for TT in Ts
            pr_branch_exists(eos, pp, TT) || (n_missing += 1)
            r = pr_density(eos, pp, TT)
            rho_min = min(rho_min, r); rho_max = max(rho_max, r)
            if isfinite(prev)
                ratio = max(prev/r, r/prev)
                if ratio > worst_ratio
                    worst_ratio = ratio; worst_at = (pp, TT)
                end
            end
            prev = r
        end
    end

    @info """Peng-Robinson $(_pr_is_liquid(eos) ? "liquid" : "vapour") branch over the envelope
    pressure      : $(ps[1]/1e5) - $(ps[end]/1e5) bar
    temperature   : $(Ts[1]) - $(Ts[end]) K
    density       : $(round(rho_min, sigdigits=5)) - $(round(rho_max, sigdigits=5)) kg/m^3
    branch absent : $(n_missing) of $(np*nT) nodes (single root - no distinct branch)
    largest step  : $(round(worst_ratio, digits=3))x between adjacent T nodes\
$(worst_ratio > 1.5 ? "  <-- at p = $(round(worst_at[1]/1e5, digits=3)) bar, T = $(round(worst_at[2], digits=3)) K" : "")"""

    return (n_missing=n_missing, worst_ratio=worst_ratio, worst_at=worst_at,
            rho_min=rho_min, rho_max=rho_max)
end

export pr_table_report
