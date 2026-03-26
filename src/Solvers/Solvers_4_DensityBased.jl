export density_based!, Rusanov, HLLC, FEuler, RK2

# ============================================================
# Boundary condition treatment — overview
# ============================================================
#
# This solver uses two independent BC dispatch chains: one for the inviscid
# (Euler) flux and one for the viscous (Navier–Stokes) flux. Both chains
# dispatch primarily on bc_U (the velocity BC type). bc_T drives the heat-flux
# sub-dispatch inside the viscous Wall method. bc_p, bc_nut and all other scalar
# BCs are irrelevant to the flux computation and do not affect dispatch.
#
# ── Inviscid BC hierarchy (_apply_inviscid_bc!, dispatch on bc_U) ─────────────
#
#   bc_U::Wall
#     → Exact Euler wall flux: F = (0, p·n·A, 0).
#       Bypasses the Riemann solver entirely. No ghost-cell dissipation.
#
#   bc_U::Union{Slip, Symmetry}
#     → Same exact Euler wall flux as Wall.
#       Both conditions enforce U·n = 0, so the flux is identical.
#       The ghost-velocity functions for Slip/Symmetry (reflect normal
#       component) are defined in this file but are NOT called by this
#       solver — they remain for use by the pressure-based solvers.
#
#   bc_U::Union{PeriodicParent, Periodic}
#     → Riemann solver with the partner cell as the right state. No ghost.
#
#   bc_U (fallback — Dirichlet, DirichletFunction, Zerogradient, …)
#     → Computes ghost velocity first, then checks the face normal velocity:
#         un_face = 0.5*(UL + UR)·n  =  U_prescribed·n   (exact arithmetic)
#       If |un_face| ≤ 1e-10: exact wall flux (catches Dirichlet(:wall,0)
#         used instead of Wall(:wall,0); no warp divergence since all faces
#         on a given patch share the same prescribed velocity).
#       Otherwise: Riemann solver (inlets, outlets, far-field).
#
# ── Viscous BC hierarchy (_apply_viscous_bc!, dispatch on bc_U then bc_T) ─────
#
#   bc_U::Wall
#     → Stress: local two-point gradient (U_wall - U_cell)/δ ⊗ n — orthogonal,
#         accurate, avoids CFL blowup from thin cell gradient amplification.
#       Heat flux: dispatches on bc_T —
#         bc_T::FixedTemperature  →  κ*(T_wall  - T_cell)/δ   (isothermal)
#         bc_T::Dirichlet         →  κ*(T_value - T_cell)/δ   (isothermal)
#         bc_T::AbstractNeumann   →  0                         (adiabatic)
#         bc_T::AbstractPhysical  →  0   (Wall/Slip/Symmetry on T = adiabatic)
#         bc_T (fallback)         →  κ*(∇T·n)  (cell-gradient projection)
#
#   bc_U::Union{Slip, Symmetry}
#     → Returns nothing. Free-slip means zero tangential shear; the symmetry
#       plane is also adiabatic by definition. bc_T is ignored entirely.
#
#   bc_U::Union{PeriodicParent, Periodic}
#     → Two-sided face-averaged gradients, consistent with internal faces.
#
#   bc_U (fallback)
#     → Cell-centred gradient projected onto face normal + bc_T heat flux.
#
# ── Scalar vs. vector field BCs: key differences ──────────────────────────────
#
#   Vector field U: bc_U type is the sole dispatch key for BOTH the inviscid
#     and viscous BC chains. Using Wall gives the exact wall treatment;
#     using Dirichlet with a zero or tangential velocity gives the same result
#     via the fallback impermeability check.
#
#   Scalar field p: bc_p enters only the Riemann-solver path (ghost pressure).
#     For Wall/Slip/Symmetry patches the Riemann solver is bypassed, so bc_p
#     has no effect on the flux. For inlet/outlet patches bc_p sets the ghost
#     pressure for the Riemann solve. Recommended types: Dirichlet (inlet, if
#     prescribed), Zerogradient (outlet or wall).
#
#   Scalar field T: bc_T selects the wall heat-flux formula in the viscous BC
#     when bc_U::Wall. Using Wall(:patch) gives adiabatic (zero heat flux).
#     Using Dirichlet(:patch, T_val) or FixedTemperature gives isothermal
#     (two-point (T_wall-T_cell)/δ). For Slip/Symmetry patches bc_T is
#     ignored (viscous flux is zero regardless). For inlet/outlet patches bc_T
#     enters the ghost temperature for the Riemann solve.
#
#   Scalar field nut: not used by either flux chain. Affects mueff/kappa_eff
#     via turbulence! → update_nueff! → mueff = ρ*nueff. Use Zerogradient
#     at inlets/outlets, Wall(:wall, 0.0) or Dirichlet(:wall, 0.0) at walls
#     (both give zero eddy viscosity at the wall face; Dirichlet is explicit).
#
# ============================================================
# Flux scheme selector types
# ============================================================

"""User-facing flux scheme type — selects the Rusanov (Local Lax-Friedrichs) flux."""
struct Rusanov end

"""User-facing flux scheme type — selects the HLLC (Harten-Lax-van Leer Contact) flux."""
struct HLLC end

# ============================================================
# Time stepping selector types
# ============================================================

"""Forward Euler (1st order) explicit time stepping for the density-based solver."""
struct FEuler end

"""
SSP-RK2 / Heun's method (2nd order) explicit time stepping for the density-based solver.

Computes two Euler stages and averages:
  W^(1)     = W^n   - (dt/V)*R(W^n)
  W^(2)     = W^(1) - (dt/V)*R(W^(1))
  W^{n+1}   = 0.5*(W^n + W^(2))
"""
struct RK2 end

# ============================================================
# Workspace struct
# ============================================================

struct DensityBasedWorkspace{SF<:ScalarField, VF<:VectorField, V<:AbstractVector}
    rhoU::VF        # conservative momentum ρU
    rhoE::SF        # conservative total energy ρE
    res_rho::SF     # density residual accumulator
    res_rhoUx::SF   # x-momentum residual accumulator
    res_rhoUy::SF   # y-momentum residual accumulator
    res_rhoUz::SF   # z-momentum residual accumulator
    res_rhoE::SF    # energy residual accumulator
    Mach::SF        # Mach number field
    dt_cell::V      # per-cell adaptive time step
    nu_eff::V       # cell-level effective kinematic viscosity (nu + nut) for diffusive CFL
    # RK2 stage storage (W^n saved before stage-1 Euler update)
    rho_0::V        # ρ   at start of time step
    rhoUx_0::V      # ρUx at start of time step
    rhoUy_0::V      # ρUy at start of time step
    rhoUz_0::V      # ρUz at start of time step
    rhoE_0::V       # ρE  at start of time step
end

# ============================================================
# Ghost state functions for boundary flux computation
# ============================================================

# --- Velocity ghost state ---

@inline function ghost_velocity(
    bc::Wall, U_int::SV, n::SV
) where {TF, SV<:SVector{3,TF}}
    # Wall: U_face = U_wall → ghost = 2*U_wall - U_int
    U_wall = SV(TF(bc.value[1]), TF(bc.value[2]), TF(bc.value[3]))
    2*U_wall - U_int
end

@inline function ghost_velocity(
    bc::Symmetry, U_int::SV, n::SV
) where {TF, SV<:SVector{3,TF}}
    # Symmetry: reflect normal component
    U_int - 2*(U_int ⋅ n)*n
end

@inline function ghost_velocity(
    bc::Slip, U_int::SV, n::SV
) where {TF, SV<:SVector{3,TF}}
    # Slip: reflect normal component (no penetration, free-slip)
    U_int - 2*(U_int ⋅ n)*n
end

@inline function ghost_velocity(
    bc::AbstractDirichlet, U_int::SV, n::SV
) where {TF, SV<:SVector{3,TF}}
    # Dirichlet: strong BC → ghost = 2*U_bc - U_int
    U_bc = SV(TF(bc.value[1]), TF(bc.value[2]), TF(bc.value[3]))
    2*U_bc - U_int
end

@inline function ghost_velocity(
    bc::AbstractNeumann, U_int::SV, n::SV
) where {TF, SV<:SVector{3,TF}}
    # Zero-gradient: extrapolate interior value
    U_int
end

@inline function ghost_velocity(
    bc::AbstractBoundary, U_int::SV, n::SV
) where {TF, SV<:SVector{3,TF}}
    # Fallback: extrapolate
    U_int
end

# Extended ghost functions with face/time/local-index context.
# Fallbacks delegate to the 3-arg versions above for all non-DirichletFunction BCs.
@inline ghost_velocity(bc, face, U_int::SV, n::SV, time, i) where {TF, SV<:SVector{3,TF}} =
    ghost_velocity(bc, U_int, n)

@inline function ghost_velocity(
    bc::DirichletFunction, face, U_int::SV, n::SV, time, i
) where {TF, SV<:SVector{3,TF}}
    U_bc = bc.value(face.centre, TF(time), i)
    SV(TF(U_bc[1]), TF(U_bc[2]), TF(U_bc[3])) * 2 - U_int
end

@inline ghost_pressure(bc, face, p_int::TF, n, time, i) where TF =
    ghost_pressure(bc, p_int, n)

@inline function ghost_pressure(bc::DirichletFunction, face, p_int::TF, n, time, i) where TF
    2*TF(bc.value(face.centre, TF(time), i)) - p_int
end

@inline ghost_temperature(bc, face, T_int::TF, time, i) where TF =
    ghost_temperature(bc, T_int)

@inline function ghost_temperature(
    bc::DirichletFunction, face, T_int::TF, time, i
) where TF
    # DirichletFunction on T: value function returns T directly
    TF(bc.value(face.centre, TF(time), i))
end

# --- Pressure ghost state ---

@inline function ghost_pressure(bc::Dirichlet, p_int::TF, n) where TF
    # Strong Dirichlet: p_face = p_bc → ghost = 2*p_bc - p_int
    2*TF(bc.value) - p_int
end

@inline function ghost_pressure(bc::AbstractNeumann, p_int::TF, n) where TF
    # Zero-gradient: extrapolate
    p_int
end

@inline function ghost_pressure(bc::AbstractPhysicalConstraint, p_int::TF, n) where TF
    # Wall/Symmetry/Slip: zero normal gradient
    p_int
end

@inline function ghost_pressure(bc::AbstractBoundary, p_int::TF, n) where TF
    # Fallback: extrapolate
    p_int
end

# --- Temperature ghost state (from T BC) ---

@inline function ghost_temperature(
    bc::FixedTemperature, T_int::TF
) where TF
    # Isothermal wall: set ghost = T_wall directly.
    # The 2*T_wall - T_int extrapolation is unstable at high Mach numbers
    # because stagnation T_int >> 2*T_wall near the stagnation point.
    # Using ghost = T_wall gives T_face = 0.5*(T_int+T_wall) which is
    # always positive and converges to T_wall as T_int → T_wall.
    TF(bc.value.T)
end

@inline function ghost_temperature(
    bc::Dirichlet, T_int::TF
) where TF
    # Dirichlet on T: bc.value is the temperature directly.
    # Use clamped approach (ghost = T_bc) for stability at high Mach
    # where T_int >> 2*T_bc near stagnation points.
    TF(bc.value)
end

@inline function ghost_temperature(
    bc::AbstractBoundary, T_int::TF
) where TF
    # Fallback (Zerogradient, Symmetry, Slip, Wall, etc.): extrapolate
    T_int
end

# ============================================================
# Rusanov (Local Lax-Friedrichs) flux
# ============================================================

@inline function rusanov_flux(
    UL::SV, UR::SV,
    pL::TF, pR::TF,
    rhoL::TF, rhoR::TF,
    normal::SV, area::TF, gamma::TF
) where {TF, SV<:SVector{3,TF}}
    # Normal velocities
    unL = UL ⋅ normal
    unR = UR ⋅ normal

    # Sound speeds
    aL = sqrt(gamma * pL / rhoL)
    aR = sqrt(gamma * pR / rhoR)

    # Local Lax-Friedrichs dissipation coefficient
    lambda = max(abs(unL) + aL, abs(unR) + aR)

    # Conservative energy variables: ρE = p/(γ-1) + 0.5*ρ|U|²
    gm1 = gamma - one(TF)
    rhoEL = pL/gm1 + TF(0.5)*rhoL*(UL ⋅ UL)
    rhoER = pR/gm1 + TF(0.5)*rhoR*(UR ⋅ UR)

    # Total enthalpy per unit mass: H = (ρE + p)/ρ
    HL = (rhoEL + pL) / rhoL
    HR = (rhoER + pR) / rhoR

    # Conservative momentum variables
    rhoUxL = rhoL*UL[1]; rhoUxR = rhoR*UR[1]
    rhoUyL = rhoL*UL[2]; rhoUyR = rhoR*UR[2]
    rhoUzL = rhoL*UL[3]; rhoUzR = rhoR*UR[3]

    # Inviscid fluxes F(W)·n at L and R
    # Mass flux
    FL_rho = rhoL*unL
    FR_rho = rhoR*unR

    # Momentum flux
    FL_rhoUx = rhoL*UL[1]*unL + pL*normal[1]
    FR_rhoUx = rhoR*UR[1]*unR + pR*normal[1]

    FL_rhoUy = rhoL*UL[2]*unL + pL*normal[2]
    FR_rhoUy = rhoR*UR[2]*unR + pR*normal[2]

    FL_rhoUz = rhoL*UL[3]*unL + pL*normal[3]
    FR_rhoUz = rhoR*UR[3]*unR + pR*normal[3]

    # Energy flux: ρH*u_n
    FL_rhoE = rhoL*HL*unL
    FR_rhoE = rhoR*HR*unR

    # Rusanov flux (×area): 0.5*(F_L+F_R) - 0.5*λ*(W_R-W_L)
    half = TF(0.5)
    F_rho  = area*(half*(FL_rho  + FR_rho)  - half*lambda*(rhoR   - rhoL))
    F_rhoUx = area*(half*(FL_rhoUx + FR_rhoUx) - half*lambda*(rhoUxR - rhoUxL))
    F_rhoUy = area*(half*(FL_rhoUy + FR_rhoUy) - half*lambda*(rhoUyR - rhoUyL))
    F_rhoUz = area*(half*(FL_rhoUz + FR_rhoUz) - half*lambda*(rhoUzR - rhoUzL))
    F_rhoE = area*(half*(FL_rhoE  + FR_rhoE)  - half*lambda*(rhoER  - rhoEL))

    return F_rho, F_rhoUx, F_rhoUy, F_rhoUz, F_rhoE
end

# ============================================================
# HLLC (Harten-Lax-van Leer Contact) flux
# ============================================================

@inline function hllc_flux(
    UL::SV, UR::SV,
    pL::TF, pR::TF,
    rhoL::TF, rhoR::TF,
    normal::SV, area::TF, gamma::TF
) where {TF, SV<:SVector{3,TF}}
    gm1 = gamma - one(TF)

    # Normal velocities
    unL = UL ⋅ normal
    unR = UR ⋅ normal

    # Sound speeds
    aL = sqrt(gamma * pL / rhoL)
    aR = sqrt(gamma * pR / rhoR)

    # Conservative total energy per unit volume
    rhoEL = pL/gm1 + TF(0.5)*rhoL*(UL ⋅ UL)
    rhoER = pR/gm1 + TF(0.5)*rhoR*(UR ⋅ UR)

    # Wave speed estimates (Einfeldt bounds)
    SL = min(unL - aL, unR - aR)
    SR = max(unL + aL, unR + aR)

    # Contact wave speed S* (from Rankine-Hugoniot conditions)
    Sstar = (pR - pL + rhoL*unL*(SL - unL) - rhoR*unR*(SR - unR)) /
            (rhoL*(SL - unL) - rhoR*(SR - unR))

    # Physical flux vectors F(W)·n
    HL = (rhoEL + pL) / rhoL
    HR = (rhoER + pR) / rhoR

    FL_rho   = rhoL*unL
    FL_rhoUx = rhoL*UL[1]*unL + pL*normal[1]
    FL_rhoUy = rhoL*UL[2]*unL + pL*normal[2]
    FL_rhoUz = rhoL*UL[3]*unL + pL*normal[3]
    FL_rhoE  = rhoL*HL*unL

    FR_rho   = rhoR*unR
    FR_rhoUx = rhoR*UR[1]*unR + pR*normal[1]
    FR_rhoUy = rhoR*UR[2]*unR + pR*normal[2]
    FR_rhoUz = rhoR*UR[3]*unR + pR*normal[3]
    FR_rhoE  = rhoR*HR*unR

    # HLLC star states (contact-preserving correction)
    # Left star: W* = ρK*(SL-unL)/(SL-S*) * [1, UK + (S*-unK)*n, EK/ρK + (S*-unK)*(S* + pK/(ρK*(SK-unK)))]
    coefL   = rhoL * (SL - unL) / (SL - Sstar)
    WL_rho  = coefL
    WL_rhoUx = coefL * (UL[1] + (Sstar - unL)*normal[1])
    WL_rhoUy = coefL * (UL[2] + (Sstar - unL)*normal[2])
    WL_rhoUz = coefL * (UL[3] + (Sstar - unL)*normal[3])
    WL_rhoE  = coefL * (rhoEL/rhoL + (Sstar - unL)*(Sstar + pL/(rhoL*(SL - unL))))

    coefR   = rhoR * (SR - unR) / (SR - Sstar)
    WR_rho  = coefR
    WR_rhoUx = coefR * (UR[1] + (Sstar - unR)*normal[1])
    WR_rhoUy = coefR * (UR[2] + (Sstar - unR)*normal[2])
    WR_rhoUz = coefR * (UR[3] + (Sstar - unR)*normal[3])
    WR_rhoE  = coefR * (rhoER/rhoR + (Sstar - unR)*(Sstar + pR/(rhoR*(SR - unR))))

    # Flux region selection (Toro, §10.4)
    if SL >= zero(TF)
        # Entirely supersonic in +n direction
        F_rho   = area*FL_rho
        F_rhoUx = area*FL_rhoUx
        F_rhoUy = area*FL_rhoUy
        F_rhoUz = area*FL_rhoUz
        F_rhoE  = area*FL_rhoE
    elseif Sstar >= zero(TF)
        # Left star region
        F_rho   = area*(FL_rho   + SL*(WL_rho   - rhoL))
        F_rhoUx = area*(FL_rhoUx + SL*(WL_rhoUx - rhoL*UL[1]))
        F_rhoUy = area*(FL_rhoUy + SL*(WL_rhoUy - rhoL*UL[2]))
        F_rhoUz = area*(FL_rhoUz + SL*(WL_rhoUz - rhoL*UL[3]))
        F_rhoE  = area*(FL_rhoE  + SL*(WL_rhoE  - rhoEL))
    elseif SR >= zero(TF)
        # Right star region
        F_rho   = area*(FR_rho   + SR*(WR_rho   - rhoR))
        F_rhoUx = area*(FR_rhoUx + SR*(WR_rhoUx - rhoR*UR[1]))
        F_rhoUy = area*(FR_rhoUy + SR*(WR_rhoUy - rhoR*UR[2]))
        F_rhoUz = area*(FR_rhoUz + SR*(WR_rhoUz - rhoR*UR[3]))
        F_rhoE  = area*(FR_rhoE  + SR*(WR_rhoE  - rhoER))
    else
        # Entirely supersonic in -n direction
        F_rho   = area*FR_rho
        F_rhoUx = area*FR_rhoUx
        F_rhoUy = area*FR_rhoUy
        F_rhoUz = area*FR_rhoUz
        F_rhoE  = area*FR_rhoE
    end

    return F_rho, F_rhoUx, F_rhoUy, F_rhoUz, F_rhoE
end

# Dispatch: select flux function based on user-chosen scheme type
@inline compute_inviscid_flux(::Rusanov, args...) = rusanov_flux(args...)
@inline compute_inviscid_flux(::HLLC,    args...) = hllc_flux(args...)

# ============================================================
# Kernels
# ============================================================

# Primitive → conservative
@kernel function _prim_to_cons!(rhoU, rhoE, rho, U, p, fluid)
    i = @index(Global)

    @uniform gamma = fluid.gamma.values

    @inbounds begin
        rho_i = rho[i]
        Ui = U[i]
        pi = p[i]

        rhoU.x[i] = rho_i * Ui[1]
        rhoU.y[i] = rho_i * Ui[2]
        rhoU.z[i] = rho_i * Ui[3]

        gm1 = gamma - one(gamma)
        KE = Ui ⋅ Ui * oftype(rho_i, 0.5)
        rhoE.values[i] = pi/gm1 + rho_i*KE
    end
end

# Initialize density from p and T
@kernel function _init_rho!(rho, p, T, fluid)
    i = @index(Global)

    @uniform R_gas = fluid.R.values

    @inbounds begin
        rho.values[i] = p[i] / (R_gas * T[i])
    end
end

# Zero all residuals
@kernel function _zero_residuals!(res_rho, res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE)
    i = @index(Global)

    @inbounds begin
        res_rho.values[i]  = zero(eltype(res_rho.values))
        res_rhoUx.values[i] = zero(eltype(res_rhoUx.values))
        res_rhoUy.values[i] = zero(eltype(res_rhoUy.values))
        res_rhoUz.values[i] = zero(eltype(res_rhoUz.values))
        res_rhoE.values[i] = zero(eltype(res_rhoE.values))
    end
end

# Inviscid flux at internal faces — cell-based loop, no atomics
@kernel function _inviscid_flux_internal!(
    flux_scheme,
    res_rho, res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE,
    rho, U, p, mesh, fluid
)
    i = @index(Global)

    @uniform begin
        (; cells, faces, cell_faces, cell_nsign) = mesh
        gamma = fluid.gamma.values
    end

    @inbounds begin
        (; volume, faces_range) = cells[i]

        TF = eltype(rho.values)
        gamma_tf = TF(gamma)

        acc_rho  = zero(TF)
        acc_rhoUx = zero(TF)
        acc_rhoUy = zero(TF)
        acc_rhoUz = zero(TF)
        acc_rhoE = zero(TF)

        for fi ∈ faces_range
            fID = cell_faces[fi]

            nsign = TF(cell_nsign[fi])
            (; area, normal, ownerCells) = faces[fID]

            cL = ownerCells[1]
            cR = ownerCells[2]

            # Left and right primitive states (always L=ownerCells[1], R=ownerCells[2])
            rhoL = TF(rho[cL])
            UL   = U[cL]
            pL   = TF(p[cL])

            rhoR = TF(rho[cR])
            UR   = U[cR]
            pR   = TF(p[cR])

            F_rho, F_rhoUx, F_rhoUy, F_rhoUz, F_rhoE = compute_inviscid_flux(
                flux_scheme, UL, UR, pL, pR, rhoL, rhoR, normal, area, gamma_tf
            )

            # nsign: +1 if cell i is left (outward), -1 if right (inward)
            acc_rho   += nsign * F_rho
            acc_rhoUx += nsign * F_rhoUx
            acc_rhoUy += nsign * F_rhoUy
            acc_rhoUz += nsign * F_rhoUz
            acc_rhoE  += nsign * F_rhoE
        end

        res_rho.values[i]  += acc_rho
        res_rhoUx.values[i] += acc_rhoUx
        res_rhoUy.values[i] += acc_rhoUy
        res_rhoUz.values[i] += acc_rhoUz
        res_rhoE.values[i] += acc_rhoE
    end
end

# Inviscid flux at boundary faces — face-based loop, uses atomics
@kernel function _inviscid_bc_flux!(
    flux_scheme, U_BCs, p_BCs, T_BCs,
    res_rho, res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE,
    rho, U, p, T, mesh, fluid, time
)
    fID = @index(Global)

    @inbounds _inviscid_bc_dispatch!(
        flux_scheme, U_BCs, p_BCs, T_BCs,
        res_rho, res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE,
        rho, U, p, T, mesh, fluid, fID, time
    )
end

# Generated dispatch over boundary patches (compile-time loop unrolling)
@generated function _inviscid_bc_dispatch!(
    flux_scheme, U_BCs, p_BCs, T_BCs,
    res_rho, res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE,
    rho, U, p, T, mesh, fluid, fID, time
)
    n = length(U_BCs.parameters)
    exprs = []
    for i ∈ 1:n
        ex = quote
            bc_U_i = U_BCs[$i]
            bc_p_i = p_BCs[$i]
            bc_T_i = T_BCs[$i]
            (; start, stop) = bc_U_i.IDs_range
            if start <= fID <= stop
                _apply_inviscid_bc!(
                    flux_scheme, bc_U_i, bc_p_i, bc_T_i,
                    res_rho, res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE,
                    rho, U, p, T, mesh, fluid, fID, time
                )
            end
        end
        push!(exprs, ex)
    end
    quote
        @inbounds begin
            $(exprs...)
        end
        nothing
    end
end

# ── Impermeable wall BCs: exact Euler flux, bypassing the Riemann solver ────────
#
# At any impermeable boundary (Wall, Slip, Symmetry) the no-penetration condition
# enforces U·n = 0 at the face. The exact inviscid (Euler) flux therefore reduces to:
#
#   F_mass     = ρ (U·n) A  =  0
#   F_momentum = (ρ U(U·n) + p n) A  =  p n A     (pure pressure force)
#   F_energy   = ρ H (U·n) A  =  0
#
# Using the Riemann solver with ghost cells at these boundaries is incorrect and
# dangerous. For a no-slip wall with U_R = -U_L (the standard mirror ghost), the
# Rusanov/HLLC numerical dissipation term is:
#
#   -½ λ (ρU_R - ρU_L)  =  λ ρ U_L        (λ ≈ |U| + a, dominated by sound speed)
#
# This injects spurious tangential momentum at magnitude ~ a * ρ * |U_tang|, which
# is large near supersonic walls and causes velocity blow-up. The kinetic energy spike
# forces pressure (and hence temperature) to drop to the clamping floor to conserve
# total energy.
#
# The exact wall flux avoids all of this: zero mass, pressure-only momentum, zero energy.

@inline function _apply_inviscid_bc!(
    flux_scheme, bc_U::Wall, bc_p, bc_T,
    res_rho, res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE,
    rho, U, p, T, mesh, fluid, fID, time
)
    (; faces) = mesh
    face = faces[fID]
    (; area, normal, ownerCells) = face
    cID = ownerCells[1]

    TF  = eltype(rho.values)
    pL  = TF(p[cID])

    # F_mass = 0 (no penetration); F_energy = 0; F_momentum = p * n * area
    Atomix.@atomic res_rhoUx.values[cID] += pL * normal[1] * area
    Atomix.@atomic res_rhoUy.values[cID] += pL * normal[2] * area
    Atomix.@atomic res_rhoUz.values[cID] += pL * normal[3] * area
end

@inline function _apply_inviscid_bc!(
    flux_scheme, bc_U::Union{Slip, Symmetry}, bc_p, bc_T,
    res_rho, res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE,
    rho, U, p, T, mesh, fluid, fID, time
)
    (; faces) = mesh
    face = faces[fID]
    (; area, normal, ownerCells) = face
    cID = ownerCells[1]

    TF  = eltype(rho.values)
    pL  = TF(p[cID])

    # F_mass = 0; F_energy = 0; F_momentum = p * n * area
    Atomix.@atomic res_rhoUx.values[cID] += pL * normal[1] * area
    Atomix.@atomic res_rhoUy.values[cID] += pL * normal[2] * area
    Atomix.@atomic res_rhoUz.values[cID] += pL * normal[3] * area
end

# ── Riemann-solver BCs: open-boundary fallback ──────────────────────────────────
#
# Handles all BC types that are not explicitly matched by a more specific method
# (Dirichlet, Neumann/Zerogradient, Extrapolated, far-field, etc.).
#
# Impermeability check: the face normal velocity is 0.5*(UL + UR)·n.  For any
# prescribed-velocity BC (Dirichlet, DirichletFunction, Wall with non-zero U_wall,
# …), this equals U_prescribed·n exactly in floating-point arithmetic because
#
#     UR = 2*U_prescribed - UL  →  0.5*(UL + UR) = U_prescribed
#
# When that quantity is zero the face is impermeable regardless of which BC type
# the user chose (e.g. Dirichlet(:ramp, noflow) instead of Wall(:ramp, noflow)).
# In that case the exact wall flux is applied rather than the Riemann solver, so
# the spurious tangential-momentum dissipation described above is avoided.
#
# For genuinely open boundaries (inlets, outlets, Zerogradient with non-zero flow)
# un_face ≠ 0 and the path falls through to the Riemann solver as intended.
@inline function _apply_inviscid_bc!(
    flux_scheme, bc_U, bc_p, bc_T,
    res_rho, res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE,
    rho, U, p, T, mesh, fluid, fID, time
)
    (; faces) = mesh
    face = faces[fID]
    (; area, normal, ownerCells) = face
    cID = ownerCells[1]
    i   = fID - bc_U.IDs_range.start + 1

    TF    = eltype(rho.values)
    gamma = TF(fluid.gamma.values)
    R_gas = TF(fluid.R.values)

    UL = U[cID]
    pL = TF(p[cID])
    TL = TF(T[cID])

    # Ghost velocity (must be computed before the impermeability check)
    UR = ghost_velocity(bc_U, face, UL, normal, time, i)

    # Face normal velocity from the two-state average.  For any prescribed-velocity
    # BC this equals the prescribed velocity's normal component exactly.
    un_face = TF(0.5) * ((UL + UR) ⋅ normal)

    if abs(un_face) <= TF(1e-10)
        # Impermeable face: exact wall flux (zero mass, pressure momentum, zero energy).
        # This catches Dirichlet / DirichletFunction BCs where the user specifies a
        # zero or purely tangential velocity instead of using Wall/Slip/Symmetry.
        Atomix.@atomic res_rhoUx.values[cID] += pL * normal[1] * area
        Atomix.@atomic res_rhoUy.values[cID] += pL * normal[2] * area
        Atomix.@atomic res_rhoUz.values[cID] += pL * normal[3] * area
        return
    end

    # Non-impermeable face: Riemann solver (inlets, outlets, far-field, …)
    rhoL = TF(rho[cID])
    pR   = ghost_pressure(bc_p, face, pL, normal, time, i)
    TR   = ghost_temperature(bc_T, face, TL, time, i)
    pR   = max(pR,   TF(1e-10))
    TR   = max(TR,   TF(1e-10))
    rhoR = max(pR / (R_gas * TR), TF(1e-10))

    F_rho, F_rhoUx, F_rhoUy, F_rhoUz, F_rhoE = compute_inviscid_flux(
        flux_scheme, UL, UR, pL, pR, rhoL, rhoR, normal, area, gamma
    )

    Atomix.@atomic res_rho.values[cID]   += F_rho
    Atomix.@atomic res_rhoUx.values[cID] += F_rhoUx
    Atomix.@atomic res_rhoUy.values[cID] += F_rhoUy
    Atomix.@atomic res_rhoUz.values[cID] += F_rhoUz
    Atomix.@atomic res_rhoE.values[cID]  += F_rhoE
end

# Periodic BC: use the partner cell's actual primitive state as the "right" state.
# Both PeriodicParent and Periodic are handled identically — each patch updates its
# own owner cell, together ensuring full conservation across the periodic interface.
@inline function _apply_inviscid_bc!(
    flux_scheme, bc_U::Union{PeriodicParent, Periodic}, bc_p, bc_T,
    res_rho, res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE,
    rho, U, p, T, mesh, fluid, fID, time
)
    (; faces) = mesh
    face = faces[fID]
    (; area, normal, ownerCells) = face
    cID = ownerCells[1]

    i    = fID - bc_U.IDs_range.start + 1
    pfID = bc_U.value.face_map[i]   # partner face ID
    pcID = faces[pfID].ownerCells[1] # partner cell ID

    TF    = eltype(rho.values)
    gamma = TF(fluid.gamma.values)

    rhoL = TF(rho[cID]);  UL = U[cID];  pL = TF(p[cID])
    rhoR = TF(rho[pcID]); UR = U[pcID]; pR = TF(p[pcID])

    F_rho, F_rhoUx, F_rhoUy, F_rhoUz, F_rhoE = compute_inviscid_flux(
        flux_scheme, UL, UR, pL, pR, rhoL, rhoR, normal, area, gamma
    )

    Atomix.@atomic res_rho.values[cID]   += F_rho
    Atomix.@atomic res_rhoUx.values[cID] += F_rhoUx
    Atomix.@atomic res_rhoUy.values[cID] += F_rhoUy
    Atomix.@atomic res_rhoUz.values[cID] += F_rhoUz
    Atomix.@atomic res_rhoE.values[cID]  += F_rhoE
end

# CFL-limited time step per cell — combined convective + diffusive restriction.
# lambda = |U| + a + nu_eff/dx so that:
#   convective regime (small nu_eff): dt ≈ CFL*dx/(|U|+a)
#   diffusive regime (large nu_eff):  dt ≈ CFL*dx²/nu_eff
@kernel function _compute_dt_cell!(dt_cell, rho, U, p, cells, fluid, cfl, dim_exp, nu_eff)
    i = @index(Global)

    @uniform gamma = fluid.gamma.values

    @inbounds begin
        TF = eltype(rho.values)
        rho_i    = TF(rho[i])
        Ui       = U[i]
        pi       = TF(p[i])
        Vi       = TF(cells[i].volume)
        nu_eff_i = TF(nu_eff[i])

        ai     = sqrt(TF(gamma) * pi / rho_i)
        Umag   = sqrt(Ui ⋅ Ui)
        dx     = Vi^TF(dim_exp)
        lambda = Umag + ai + nu_eff_i / (dx + TF(1e-30))

        dt_cell[i] = TF(cfl) * dx / (lambda + TF(1e-30))
    end
end

# Compute cell-level effective viscosity: nu_eff[i] = nu_mol + nut[i]
@kernel function _compute_nu_eff_cell!(nu_eff, nu_mol, nut)
    i = @index(Global)
    @inbounds nu_eff[i] = nu_mol + nut.values[i]
end

# Populate cell-level nu_eff following the same dispatch pattern as update_nueff!
# (Solvers_0_functions.jl). turb_model = model.turbulence (the struct created by the
# LES/RANS functor, e.g. Smagorinsky, KOmega, Laminar — NOT the initialised SmagorinskyModel).
# Turbulent models have a nut::ScalarField field; Laminar does not.
function update_nu_eff_cell!(nu_eff, nu_mol, turb_model, backend, workgroup, n_cells)
    if typeof(turb_model) <: Laminar
        @. nu_eff = nu_mol
    else
        kernel! = _compute_nu_eff_cell!(_setup(backend, workgroup, n_cells)...)
        kernel!(nu_eff, nu_mol, turb_model.nut)
    end
end

# Forward Euler update of conservative variables
@kernel function _forward_euler!(rho, rhoU, rhoE, res_rho, res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE, cells, dt)
    i = @index(Global)

    @inbounds begin
        TF = eltype(rho.values)
        V  = TF(cells[i].volume)
        dt_tf = TF(dt)
        factor = dt_tf / V

        rho.values[i]   -= factor * res_rho.values[i]
        rhoU.x[i]       -= factor * res_rhoUx.values[i]
        rhoU.y[i]       -= factor * res_rhoUy.values[i]
        rhoU.z[i]       -= factor * res_rhoUz.values[i]
        rhoE.values[i]  -= factor * res_rhoE.values[i]

        # Clamp density to positive
        rho.values[i] = max(rho.values[i], TF(1e-10))
    end
end

# Save conservative state W^n (for RK2 averaging)
@kernel function _save_conservative!(rho_0, rhoUx_0, rhoUy_0, rhoUz_0, rhoE_0, rho, rhoU, rhoE)
    i = @index(Global)

    @inbounds begin
        rho_0[i]   = rho.values[i]
        rhoUx_0[i] = rhoU.x[i]
        rhoUy_0[i] = rhoU.y[i]
        rhoUz_0[i] = rhoU.z[i]
        rhoE_0[i]  = rhoE.values[i]
    end
end

# RK2 average: W^{n+1} = 0.5*(W^n + W^(2)), with density clamping
@kernel function _rk2_average!(rho, rhoU, rhoE, rho_0, rhoUx_0, rhoUy_0, rhoUz_0, rhoE_0)
    i = @index(Global)

    @inbounds begin
        TF = eltype(rho.values)
        rho_new = TF(0.5) * (rho_0[i] + rho.values[i])
        rho.values[i] = max(rho_new, TF(1e-10))
        rhoU.x[i]     = TF(0.5) * (rhoUx_0[i] + rhoU.x[i])
        rhoU.y[i]     = TF(0.5) * (rhoUy_0[i] + rhoU.y[i])
        rhoU.z[i]     = TF(0.5) * (rhoUz_0[i] + rhoU.z[i])
        rhoE.values[i] = TF(0.5) * (rhoE_0[i] + rhoE.values[i])
    end
end

# Conservative → primitive recovery
@kernel function _cons_to_prim!(U, p, T, Mach, rho, rhoU, rhoE, cells, fluid)
    i = @index(Global)

    @uniform begin
        gamma = fluid.gamma.values
        R_gas = fluid.R.values
    end

    @inbounds begin
        TF = eltype(rho.values)
        gamma_tf = TF(gamma)
        R_gas_tf = TF(R_gas)

        rho_i  = TF(rho[i])
        rhoE_i = TF(rhoE[i])

        # Velocity
        ux = rhoU.x[i] / rho_i
        uy = rhoU.y[i] / rho_i
        uz = rhoU.z[i] / rho_i

        U.x[i] = ux
        U.y[i] = uy
        U.z[i] = uz

        # Kinetic energy per unit volume
        KE = TF(0.5) * (ux*ux + uy*uy + uz*uz)

        # Pressure: p = (γ-1)*(ρE - ρ*KE)
        p_i = (gamma_tf - one(TF)) * (rhoE_i - rho_i*KE)
        p_i = max(p_i, TF(1e-10))
        p.values[i] = p_i

        # Temperature: T = p/(ρ*R)
        T_i = p_i / (rho_i * R_gas_tf)
        T_i = max(T_i, TF(1e-10))
        T.values[i] = T_i

        # Mach number
        a = sqrt(gamma_tf * p_i / rho_i)
        Mach.values[i] = sqrt(ux*ux + uy*uy + uz*uz) / (a + TF(1e-30))
    end
end

# ============================================================
# Viscous flux kernels
# ============================================================

# Internal faces — cell-based loop, same pattern as _rusanov_flux_internal!
@kernel function _viscous_flux_internal!(
    res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE,
    U, gradU, gradT, mueff, kappa_eff, mesh
)
    i = @index(Global)

    @uniform begin
        (; cells, faces, cell_faces, cell_nsign) = mesh
    end

    @inbounds begin
        (; faces_range) = cells[i]

        TF = eltype(res_rhoUx.values)
        two_thirds = TF(2)/TF(3)

        acc_rhoUx = zero(TF)
        acc_rhoUy = zero(TF)
        acc_rhoUz = zero(TF)
        acc_rhoE  = zero(TF)

        for fi ∈ faces_range
            fID     = cell_faces[fi]
            nsign   = TF(cell_nsign[fi])
            (; area, normal, ownerCells) = faces[fID]

            cL = ownerCells[1]
            cR = ownerCells[2]

            # Face-averaged velocity gradient and temperature gradient
            gradU_f = TF(0.5) * (gradU[cL] + gradU[cR])
            gradT_f = TF(0.5) * (gradT[cL] + gradT[cR])

            # Velocity divergence (trace of ∇U)
            divU = gradU_f[1,1] + gradU_f[2,2] + gradU_f[3,3]

            # Viscous stress projection onto face normal: (μ*(∇U + (∇U)ᵀ - 2/3*(∇·U)I)) · n
            mueff_f = TF(mueff[fID])
            tau_n   = mueff_f * ((gradU_f + gradU_f') * normal - two_thirds * divU * normal)

            # Face-averaged velocity
            U_f = TF(0.5) * (U[cL] + U[cR])

            # Viscous energy flux: u·(τ·n) + κ*(∇T·n)
            kf       = TF(kappa_eff[fID])
            F_visc_E = (U_f ⋅ tau_n) + kf * (gradT_f ⋅ normal)

            # Subtract viscous contribution (RHS term → subtract from residual)
            acc_rhoUx -= nsign * tau_n[1] * area
            acc_rhoUy -= nsign * tau_n[2] * area
            acc_rhoUz -= nsign * tau_n[3] * area
            acc_rhoE  -= nsign * F_visc_E * area
        end

        res_rhoUx.values[i] += acc_rhoUx
        res_rhoUy.values[i] += acc_rhoUy
        res_rhoUz.values[i] += acc_rhoUz
        res_rhoE.values[i]  += acc_rhoE
    end
end

# ============================================================
# Wall heat flux helpers — dispatch on energy BC type
# ============================================================

# Adiabatic (Zerogradient, Extrapolated, etc.): zero heat flux through the face
@inline wall_heat_flux(
    ::AbstractNeumann, kf, gradT, T, normal, delta, cID, fID
) = zero(kf)

# Adiabatic (Wall, Slip, Symmetry on T field): zero heat flux
@inline wall_heat_flux(
    ::AbstractPhysicalConstraint, kf, gradT, T, normal, delta, cID, fID
) = zero(kf)

# Isothermal wall (FixedTemperature): two-point orthogonal normal gradient
@inline function wall_heat_flux(
    bc_T::FixedTemperature, kf, gradT, T, normal, delta, cID, fID
)
    T_wall = typeof(kf)(bc_T.value.T)
    T_cell = typeof(kf)(T[cID])
    kf * (T_wall - T_cell) / typeof(kf)(delta)
end

# Isothermal wall (Dirichlet on T): two-point orthogonal normal gradient
@inline function wall_heat_flux(
    bc_T::Dirichlet, kf, gradT, T, normal, delta, cID, fID
)
    T_wall = typeof(kf)(bc_T.value)
    T_cell = typeof(kf)(T[cID])
    kf * (T_wall - T_cell) / typeof(kf)(delta)
end

# Fallback: cell-centred gradient projected onto face normal
@inline function wall_heat_flux(
    bc_T, kf, gradT, T, normal, delta, cID, fID
)
    kf * (gradT[cID] ⋅ normal)
end

# ============================================================
# Boundary faces — face-based loop with atomics, dispatches on BC type
@kernel function _viscous_bc_flux!(
    U_BCs, T_BCs,
    res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE,
    U, Uf, gradU, gradT, T, mueff, kappa_eff, mesh
)
    fID = @index(Global)

    @inbounds _viscous_bc_dispatch!(
        U_BCs, T_BCs,
        res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE,
        U, Uf, gradU, gradT, T, mueff, kappa_eff, mesh, fID
    )
end

# Generated dispatch over boundary patches (compile-time loop unrolling)
@generated function _viscous_bc_dispatch!(
    U_BCs, T_BCs,
    res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE,
    U, Uf, gradU, gradT, T, mueff, kappa_eff, mesh, fID
)
    n = length(U_BCs.parameters)
    exprs = []
    for i ∈ 1:n
        ex = quote
            bc_U_i = U_BCs[$i]
            bc_T_i = T_BCs[$i]
            (; start, stop) = bc_U_i.IDs_range
            if start <= fID <= stop
                _apply_viscous_bc!(
                    bc_U_i, bc_T_i,
                    res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE,
                    U, Uf, gradU, gradT, T, mueff, kappa_eff, mesh, fID
                )
            end
        end
        push!(exprs, ex)
    end
    quote
        @inbounds begin
            $(exprs...)
        end
        nothing
    end
end

# Generic viscous BC (Dirichlet, Neumann, etc.)
# Uses the cell-centred gradient directly (bounded for explicit solvers; the
# 1/delta correction was removed because it amplifies stresses at thin wall cells).
# Dispatches heat flux on the T BC type: AbstractNeumann → zero, FixedTemperature/Dirichlet → (T_wall-T_cell)/delta.
@inline function _apply_viscous_bc!(
    bc_U, bc_T,
    res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE,
    U, Uf, gradU, gradT, T, mueff, kappa_eff, mesh, fID
)
    (; faces) = mesh
    face = faces[fID]
    (; area, normal, delta, ownerCells) = face
    cID = ownerCells[1]

    TF = eltype(res_rhoUx.values)
    two_thirds = TF(2)/TF(3)

    # Face velocity from correct_boundaries! (U_wall for no-slip, interpolated for others)
    Uf_face = Uf[fID]

    # Use cell-centred gradient directly (more stable than face-centred correction for
    # explicit solvers; the 1/delta factor in (Uf-Ucell)/delta amplifies stresses for
    # thin boundary-layer cells, causing CFL violations on the viscous update)
    gradU_f = gradU[cID]

    # Velocity divergence
    divU = gradU_f[1,1] + gradU_f[2,2] + gradU_f[3,3]

    # Viscous stress projection using cell-centred gradient
    mueff_f = TF(mueff[fID])
    tau_n   = mueff_f * ((gradU_f + gradU_f') * normal - two_thirds * divU * normal)

    # Heat flux: dispatch on T BC type
    kf       = TF(kappa_eff[fID])
    q_wall   = wall_heat_flux(bc_T, kf, gradT, T, normal, delta, cID, fID)

    # Viscous energy flux
    F_visc_E = (Uf_face ⋅ tau_n) + q_wall

    # Subtract viscous contribution (boundary faces are outward from cID)
    Atomix.@atomic res_rhoUx.values[cID] -= tau_n[1] * area
    Atomix.@atomic res_rhoUy.values[cID] -= tau_n[2] * area
    Atomix.@atomic res_rhoUz.values[cID] -= tau_n[3] * area
    Atomix.@atomic res_rhoE.values[cID]  -= F_visc_E * area
end

# Wall (no-slip): local two-point orthogonal normal gradient for accurate skin friction.
# The gradient tensor is reconstructed from (U_wall - U_cell)/delta along the wall normal,
# capturing the dominant wall-normal shear without amplifying errors from thin cells.
@inline function _apply_viscous_bc!(
    bc_U::Wall, bc_T,
    res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE,
    U, Uf, gradU, gradT, T, mueff, kappa_eff, mesh, fID
)
    (; faces) = mesh
    face = faces[fID]
    (; area, normal, delta, ownerCells) = face
    cID = ownerCells[1]

    TF = eltype(res_rhoUx.values)
    two_thirds = TF(2)/TF(3)

    # Known wall velocity (from BC)
    U_wall = SVector{3,TF}(TF(bc_U.value[1]), TF(bc_U.value[2]), TF(bc_U.value[3]))
    U_cell = U[cID]

    # Local two-point gradient: gradU_ij ≈ (U_wall_i - U_cell_i) * n_j / delta
    dU     = (U_wall - U_cell) / TF(delta)
    gradU_f = dU * normal'   # outer product → 3×3 SMatrix

    divU    = gradU_f[1,1] + gradU_f[2,2] + gradU_f[3,3]
    mueff_f = TF(mueff[fID])
    tau_n   = mueff_f * ((gradU_f + gradU_f') * normal - two_thirds * divU * normal)

    # Heat flux: dispatch on T BC type
    kf     = TF(kappa_eff[fID])
    q_wall = wall_heat_flux(bc_T, kf, gradT, T, normal, delta, cID, fID)

    # Energy flux: viscous work at wall velocity + heat conduction
    F_visc_E = (U_wall ⋅ tau_n) + q_wall

    Atomix.@atomic res_rhoUx.values[cID] -= tau_n[1] * area
    Atomix.@atomic res_rhoUy.values[cID] -= tau_n[2] * area
    Atomix.@atomic res_rhoUz.values[cID] -= tau_n[3] * area
    Atomix.@atomic res_rhoE.values[cID]  -= F_visc_E * area
end

# Slip/Symmetry: free-slip condition → zero tangential shear + adiabatic → no viscous flux.
@inline function _apply_viscous_bc!(
    bc_U::Union{Slip, Symmetry}, bc_T,
    res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE,
    U, Uf, gradU, gradT, T, mueff, kappa_eff, mesh, fID
)
    nothing
end

# Periodic BC viscous flux uses two-sided (face-averaged) gradients and velocity,
# consistent with the internal face treatment.
@inline function _apply_viscous_bc!(
    bc_U::Union{PeriodicParent, Periodic}, bc_T,
    res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE,
    U, Uf, gradU, gradT, T, mueff, kappa_eff, mesh, fID
)
    (; faces) = mesh
    face = faces[fID]
    (; area, normal, ownerCells) = face
    cID = ownerCells[1]

    i    = fID - bc_U.IDs_range.start + 1
    pfID = bc_U.value.face_map[i]    # partner face ID
    pcID = faces[pfID].ownerCells[1] # partner cell ID

    TF = eltype(res_rhoUx.values)
    two_thirds = TF(2)/TF(3)

    # Two-sided face-averaged gradients (consistent with internal face treatment)
    gradU_f = TF(0.5) * (gradU[cID] + gradU[pcID])
    gradT_f = TF(0.5) * (gradT[cID] + gradT[pcID])

    # Velocity divergence
    divU = gradU_f[1,1] + gradU_f[2,2] + gradU_f[3,3]

    # Viscous stress projection
    mueff_f = TF(mueff[fID])
    tau_n   = mueff_f * ((gradU_f + gradU_f') * normal - two_thirds * divU * normal)

    # Face-averaged velocity
    Uf_face = TF(0.5) * (U[cID] + U[pcID])

    # Viscous energy flux
    kf       = TF(kappa_eff[fID])
    F_visc_E = (Uf_face ⋅ tau_n) + kf * (gradT_f ⋅ normal)

    Atomix.@atomic res_rhoUx.values[cID] -= tau_n[1] * area
    Atomix.@atomic res_rhoUy.values[cID] -= tau_n[2] * area
    Atomix.@atomic res_rhoUz.values[cID] -= tau_n[3] * area
    Atomix.@atomic res_rhoE.values[cID]  -= F_visc_E * area
end

# ============================================================
# Setup and main solver loop
# ============================================================

"""
    density_based!(model, config; output=VTK())

Entry point for the explicit density-based compressible solver targeting
supersonic/hypersonic flows. Uses Rusanov (Local Lax-Friedrichs) flux with
Forward Euler time integration on conservative variables [ρ, ρU, ρE].

Dispatched automatically from `run!` when `model.fluid isa SupersonicFlow`.
"""
function density_based!(model, config; output=VTK())
    residuals = _setup_density_based(model, config; output=output)
    return residuals
end

function _setup_density_based(model, config; output=VTK())
    (; U, p, Uf, pf) = model.momentum
    (; rho, nu) = model.fluid
    mesh = model.domain
    (; hardware, runtime, schemes, boundaries) = config
    (; backend, workgroup) = hardware

    @info "Allocating DensityBasedWorkspace..."

    n_cells = length(mesh.cells)
    TF = _get_float(mesh)

    rhoU     = VectorField(mesh)
    rhoE     = ScalarField(mesh)
    res_rho  = ScalarField(mesh)
    res_rhoUx = ScalarField(mesh)
    res_rhoUy = ScalarField(mesh)
    res_rhoUz = ScalarField(mesh)
    res_rhoE = ScalarField(mesh)
    Mach     = ScalarField(mesh)
    dt_cell      = KernelAbstractions.zeros(backend, TF, n_cells)
    nu_eff_cell  = KernelAbstractions.zeros(backend, TF, n_cells)
    rho_0    = KernelAbstractions.zeros(backend, TF, n_cells)
    rhoUx_0  = KernelAbstractions.zeros(backend, TF, n_cells)
    rhoUy_0  = KernelAbstractions.zeros(backend, TF, n_cells)
    rhoUz_0  = KernelAbstractions.zeros(backend, TF, n_cells)
    rhoE_0   = KernelAbstractions.zeros(backend, TF, n_cells)

    workspace = DensityBasedWorkspace(
        rhoU, rhoE, res_rho, res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE, Mach, dt_cell,
        nu_eff_cell, rho_0, rhoUx_0, rhoUy_0, rhoUz_0, rhoE_0
    )

    @info "Allocating viscous and turbulence fields..."

    # Gradient objects: created before T is bound to temperature (T type still available)
    gradU  = Grad{schemes.U.gradient}(U)
    gradUT = T(gradU)
    Uf_s   = FaceVectorField(mesh)  # Uf for StrainRate (separate from momentum Uf)
    S      = StrainRate(gradU, gradUT, U, Uf_s)

    # Temperature gradient (T_field = model.energy.T, Tf = model.energy.Tf for face values)
    T_field = model.energy.T
    gradT   = Grad{schemes.T.gradient}(T_field)

    # Effective viscosity / conductivity on faces
    nueff     = FaceScalarField(mesh)
    mueff     = FaceScalarField(mesh)
    kappa_eff = FaceScalarField(mesh)

    # Mass flux for turbulence transport equations
    mdotf = FaceScalarField(mesh)

    # Scratch array for turbulence! prev argument
    prev = KernelAbstractions.zeros(backend, TF, n_cells)

    @info "Initialising turbulence model..."
    turbulenceModel, config = initialise(model.turbulence, model, mdotf, nothing, config)

    @info "Initialising density from p and T..."

    ndrange = n_cells
    kernel! = _init_rho!(_setup(backend, workgroup, ndrange)...)
    kernel!(rho, p, T_field, model.fluid)

    residuals = DENSITY_BASED(
        model, workspace, turbulenceModel,
        S, gradT, nueff, mueff, kappa_eff, mdotf, prev,
        config; output=output
    )
    return residuals
end

# ============================================================
# Helper: compute all residuals from current primitive state
# ============================================================

function compute_residuals!(
    workspace, flux_scheme, boundaries, model, gradU, gradT, mueff, kappa_eff,
    Uf, mesh, time, config
)
    (; res_rho, res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE) = workspace
    (; U, p) = model.momentum
    (; rho) = model.fluid
    T = model.energy.T
    (; hardware) = config
    (; backend, workgroup) = hardware
    n_cells  = length(mesh.cells)
    n_bfaces = length(mesh.boundary_cellsID)

    # Zero residuals
    kernel! = _zero_residuals!(_setup(backend, workgroup, n_cells)...)
    kernel!(res_rho, res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE)

    # Inviscid flux — internal faces
    kernel! = _inviscid_flux_internal!(_setup(backend, workgroup, n_cells)...)
    kernel!(
        flux_scheme,
        res_rho, res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE,
        rho, U, p, mesh, model.fluid
    )

    # Viscous flux — internal faces
    kernel! = _viscous_flux_internal!(_setup(backend, workgroup, n_cells)...)
    kernel!(
        res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE,
        U, gradU, gradT, mueff, kappa_eff, mesh
    )

    # Inviscid flux — boundary faces
    kernel! = _inviscid_bc_flux!(_setup(backend, workgroup, n_bfaces)...)
    kernel!(
        flux_scheme, boundaries.U, boundaries.p, boundaries.T,
        res_rho, res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE,
        rho, U, p, T, mesh, model.fluid, time
    )

    # Viscous flux — boundary faces
    kernel! = _viscous_bc_flux!(_setup(backend, workgroup, n_bfaces)...)
    kernel!(
        boundaries.U, boundaries.T,
        res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE,
        U, Uf, gradU, gradT, T, mueff, kappa_eff, mesh
    )
end

# ============================================================
# Helper: conservative → primitive recovery
# ============================================================

# Recover cell-centred primitive variables (U, p, T, Mach) from the current
# conservative state (rho, rhoU, rhoE). Must be called before any operation
# that depends on primitive fields (face interpolations, gradient updates, etc.).
function recover_primitives!(workspace, model, config)
    (; rhoU, rhoE, Mach) = workspace
    (; U, p) = model.momentum
    (; rho) = model.fluid
    T = model.energy.T
    (; hardware) = config
    (; backend, workgroup) = hardware
    n_cells = length(model.domain.cells)

    kernel! = _cons_to_prim!(_setup(backend, workgroup, n_cells)...)
    kernel!(U, p, T, Mach, rho, rhoU, rhoE, model.domain.cells, model.fluid)
end

# ============================================================
# Helper: interpolate primitive fields to faces + update mass flux
# ============================================================

# Interpolate all cell-centred primitive fields (U, p, T, rho) to faces,
# apply boundary corrections, and recompute the face mass flux mdotf = ρ*Uf·Sf.
# Must be called AFTER recover_primitives! so that U, p, T, rho are current.
function interpolate_primitive_fields!(
    model, boundaries, rhof, mdotf, time, config
)
    (; U, p, Uf, pf) = model.momentum
    (; rho) = model.fluid
    (; T, Tf) = model.energy

    interpolate!(Uf, U, config)
    correct_boundaries!(Uf, U, boundaries.U, time, config)
    interpolate!(pf, p, config)
    correct_boundaries!(pf, p, boundaries.p, time, config)
    interpolate!(Tf, T, config)
    correct_boundaries!(Tf, T, boundaries.T, time, config)

    interpolate!(rhof, rho, config)

    flux!(mdotf, Uf, config)
    @. mdotf.values *= rhof.values
end

# ============================================================
# Time stepping dispatch: step!
# ============================================================

"""
    step!(::FEuler, workspace, model, boundaries, flux_scheme,
          gradU, gradT, nueff, mueff, kappa_eff, rhof, mdotf, dt, time, config)

Single Forward-Euler update of the conserved variables using the residuals
already computed by compute_residuals!. After this call, recover_primitives!
and interpolate_primitive_fields! must be called to synchronise the primitive
and face fields with the updated conservative state.
"""
function step!(
    ::FEuler, workspace, model, boundaries, flux_scheme,
    gradU, gradT, nueff, mueff, kappa_eff, rhof, mdotf, dt, time, config
)
    (; rhoU, rhoE, res_rho, res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE) = workspace
    (; rho) = model.fluid
    (; hardware) = config
    (; backend, workgroup) = hardware
    mesh = model.domain
    n_cells = length(mesh.cells)

    kernel! = _forward_euler!(_setup(backend, workgroup, n_cells)...)
    kernel!(rho, rhoU, rhoE, res_rho, res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE, mesh.cells, dt)
end

"""
    step!(::RK2, workspace, model, boundaries, flux_scheme,
          gradU, gradT, nueff, mueff, kappa_eff, rhof, mdotf, dt, time, config)

SSP-RK2 (Heun's method) update of the conserved variables:
  W^(1)   = W^n   - (dt/V)*R(W^n)
  W^(2)   = W^(1) - (dt/V)*R(W^(1))
  W^{n+1} = 0.5*(W^n + W^(2))

R(W^n) must already be computed by compute_residuals! before calling step!.
After this call, recover_primitives! and interpolate_primitive_fields! must be
called to synchronise the primitive and face fields with W^{n+1}.
"""
function step!(
    ::RK2, workspace, model, boundaries, flux_scheme,
    gradU, gradT, nueff, mueff, kappa_eff, rhof, mdotf, dt, time, config
)
    (; rhoU, rhoE, res_rho, res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE) = workspace
    (; rho_0, rhoUx_0, rhoUy_0, rhoUz_0, rhoE_0) = workspace
    (; rho) = model.fluid
    (; U, Uf) = model.momentum
    (; T, Tf) = model.energy
    (; hardware, schemes) = config
    (; backend, workgroup) = hardware
    mesh = model.domain
    n_cells = length(mesh.cells)
    TF = _get_float(mesh)
    cp_val = TF(model.fluid.cp.values)
    Pr_val = TF(model.fluid.Pr.values)

    # --- Stage 1: save W^n, Euler update using R(W^n) already computed ---
    kernel! = _save_conservative!(_setup(backend, workgroup, n_cells)...)
    kernel!(rho_0, rhoUx_0, rhoUy_0, rhoUz_0, rhoE_0, rho, rhoU, rhoE)

    kernel! = _forward_euler!(_setup(backend, workgroup, n_cells)...)
    kernel!(rho, rhoU, rhoE, res_rho, res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE, mesh.cells, dt)

    # Recover primitives and update face fields from W^(1) so that
    # gradients and viscous coefficients are evaluated at the intermediate state.
    recover_primitives!(workspace, model, config)
    interpolate_primitive_fields!(model, boundaries, rhof, mdotf, time, config)

    # Recompute gradients and viscous transport coefficients at W^(1)
    grad!(gradT, Tf, T, boundaries.T, time, config)
    limit_gradient!(schemes.T.limiter, gradT, T, config)
    grad!(gradU, Uf, U, boundaries.U, time, config)
    limit_gradient!(schemes.U.limiter, gradU, U, config)
    @. mueff.values     = rhof.values * nueff.values
    @. kappa_eff.values = mueff.values * cp_val / Pr_val

    # --- Stage 2: compute R(W^(1)), Euler update → W^(2) ---
    compute_residuals!(
        workspace, flux_scheme, boundaries, model, gradU, gradT, mueff, kappa_eff,
        Uf, mesh, time, config
    )

    kernel! = _forward_euler!(_setup(backend, workgroup, n_cells)...)
    kernel!(rho, rhoU, rhoE, res_rho, res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE, mesh.cells, dt)

    # --- Convex average: W^{n+1} = 0.5*(W^n + W^(2)) ---
    kernel! = _rk2_average!(_setup(backend, workgroup, n_cells)...)
    kernel!(rho, rhoU, rhoE, rho_0, rhoUx_0, rhoUy_0, rhoUz_0, rhoE_0)
end

# ============================================================
# Main solver loop
# ============================================================

function DENSITY_BASED(
    model, workspace, turbulenceModel,
    S, gradT, nueff, mueff, kappa_eff, mdotf, prev,
    config; output=VTK()
)
    (; U, p, Uf, pf) = model.momentum
    (; rho, nu, R) = model.fluid
    (; T, Tf) = model.energy  # face temperature (FaceScalarField)
    mesh = model.domain
    (; schemes, runtime, hardware, boundaries) = config
    (; iterations, write_interval) = runtime
    (; backend, workgroup) = hardware

    # Flux scheme: user sets schemes = (..., flux = HLLC()) or flux = Rusanov(); default Rusanov
    flux_scheme = get(schemes, :flux, Rusanov())

    # Time stepping: user sets schemes = (..., time_stepping = RK2()); default FEuler
    time_scheme = get(schemes, :time_stepping, FEuler())

    (; rhoU, rhoE, res_rho, dt_cell) = workspace

    n_cells = length(mesh.cells)
    TF = _get_float(mesh)

    # CFL number from adaptive time stepping or default
    cfl = if !isnothing(runtime.adaptive)
        TF(runtime.adaptive.maxCo)
    else
        TF(0.5)
    end

    # Cell size exponent: 1/2 for 2D, 1/3 for 3D
    dim_exp = typeof(mesh) <: Mesh2 ? TF(0.5) : TF(0.333333)

    outputWriter = initialise_writer(output, model.domain)

    # Extract gradU from the StrainRate object (updated by turbulence! each iteration)
    (; gradU) = S

    # Fluid constants for thermal conductivity: κ_eff = μ_eff * cp / Pr
    cp_val = TF(model.fluid.cp.values)
    Pr_val = TF(model.fluid.Pr.values)
    rhof   = model.fluid.rhof  # face density (FaceScalarField in SupersonicFlow)

    @info "Initialising conservative variables from primitive fields..."

    kernel! = _prim_to_cons!(_setup(backend, workgroup, n_cells)...)
    kernel!(rhoU, rhoE, rho, U, p, model.fluid)

    time = TF(0.0)

    # Molecular viscosity (scalar constant for SupersonicFlow)
    nu_mol = TF(model.fluid.nu.values)

    @info "Initialising face fields, gradients and viscous coefficients..."

    # Ensure density is consistent with the initial p and T, then populate all
    # face fields. Uf and Tf must be filled before grad! is called.
    @. rho.values = p.values / (R.values * T.values)
    interpolate_primitive_fields!(model, boundaries, rhof, mdotf, time, config)

    # Initial effective viscosity: nueff (face kinematic) → mueff = ρ*νeff → κ_eff
    update_nueff!(nueff, nu, model.turbulence, config)
    @. mueff.values     = rhof.values * nueff.values
    @. kappa_eff.values = mueff.values * cp_val / Pr_val

    # Initial velocity and temperature gradients (Uf and Tf are now populated)
    grad!(gradU, Uf, U, boundaries.U, time, config)
    limit_gradient!(schemes.U.limiter, gradU, U, config)
    grad!(gradT, Tf, T, boundaries.T, time, config)
    limit_gradient!(schemes.T.limiter, gradT, T, config)

    # Initialise cell-level nu_eff used for the diffusive CFL restriction
    @. workspace.nu_eff = nu_mol

    # Pre-allocate residual storage
    R_rho = ones(TF, iterations)

    @info "Starting DENSITY_BASED time loop ($(typeof(time_scheme)))..."

    progress = Progress(iterations; dt=1.0, showspeed=true)

    for iteration ∈ 1:iterations

        # 1. Compute inviscid + viscous residuals R(W^n) using current primitive
        #    fields and gradients (updated at the end of the previous iteration).
        compute_residuals!(
            workspace, flux_scheme, boundaries, model, gradU, gradT, mueff, kappa_eff,
            Uf, mesh, time, config
        )

        # 2. L2 norm of density residual (recorded before the conservative update)
        rho_res = norm(res_rho.values) / sqrt(TF(n_cells))
        R_rho[iteration] = rho_res

        # 3. CFL-limited global time step (convective + diffusive restriction).
        #    workspace.nu_eff holds the cell-level effective kinematic viscosity
        #    updated at the end of the previous iteration.
        kernel! = _compute_dt_cell!(_setup(backend, workgroup, n_cells)...)
        kernel!(dt_cell, rho, U, p, mesh.cells, model.fluid, cfl, dim_exp, workspace.nu_eff)
        dt = minimum(dt_cell)

        # Update runtime dt (used by output writers)
        runtime.dt .= dt
        time += dt

        # 4. Time step: advance conservative variables W^n → W^{n+1}.
        #    For FEuler: single Euler update using R(W^n).
        #    For RK2:    two-stage SSP-RK2; stage-1 primitive/face fields are
        #                recovered internally before the stage-2 residual evaluation.
        step!(
            time_scheme, workspace, model, boundaries, flux_scheme,
            gradU, gradT, nueff, mueff, kappa_eff, rhof, mdotf, dt, time, config
        )

        # 5. Recover cell-centred primitive variables from the updated W^{n+1}.
        recover_primitives!(workspace, model, config)

        # 6. Interpolate primitive fields to faces and recompute the mass flux.
        interpolate_primitive_fields!(model, boundaries, rhof, mdotf, time, config)

        # 7. Recompute temperature gradient from the updated Tf.
        grad!(gradT, Tf, T, boundaries.T, time, config)
        limit_gradient!(schemes.T.limiter, gradT, T, config)

        # 8. Turbulence update.
        #    turbulence! internally recomputes grad(U) and the strain-rate tensor S
        #    from the updated Uf, which is required to form the eddy-viscosity nut.
        turbulence!(turbulenceModel, model, S, prev, time, config)

        # 9. Recompute effective viscosity on faces (nueff = nu + nut) and the
        #    dynamic viscosity/thermal conductivity fields used in the viscous flux.
        update_nueff!(nueff, nu, model.turbulence, config)
        @. mueff.values     = rhof.values * nueff.values
        @. kappa_eff.values = mueff.values * cp_val / Pr_val

        # 10. Update cell-level effective kinematic viscosity for the diffusive
        #     CFL restriction at the next iteration.
        update_nu_eff_cell!(workspace.nu_eff, nu_mol, model.turbulence, backend, workgroup, n_cells)

        # 11. Progress and convergence check
        ProgressMeter.next!(
            progress, showvalues = [
                (:iter, iteration),
                (:dt, dt),
                (:rho_residual, rho_res),
                turbulenceModel.state.residuals...
            ]
        )

        # 12. Write output at specified interval
        if iteration % write_interval + signbit(write_interval) == 0
            save_output(model, outputWriter, iteration, time, config)
        end

    end # time loop

    return (rho=R_rho,)
end
