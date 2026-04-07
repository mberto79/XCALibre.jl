export godunov!, Rusanov, HLLC, FEuler, RK2, MUSCL, VanLeer, MinMod, Superbee

# ============================================================
# Boundary condition dispatch overview
# ============================================================
#
# Two independent BC dispatch chains, both keyed on bc_U:
#
# Inviscid (_apply_inviscid_bc!):
#   Wall / Slip / Symmetry  → exact Euler wall flux: F=(0, p·n·A, 0)
#   PeriodicParent/Periodic → Riemann solve with partner cell state
#   Outlet                  → outflow: UR=UL (zero-gradient); backflow: UR=0 (stagnant ghost)
#   AbstractDirichlet       → ghost UR=2*U_bc-UL; tangential prescription → wall flux
#   fallback                → ghost, then impermeability check → wall flux or Riemann solve
#
# Viscous (_apply_viscous_bc!, bc_U then bc_T):
#   Wall     → τ from two-point gradient (U_wall-U_cell)/δ⊗n; bc_T selects heat flux:
#                FixedTemperature/Dirichlet → κ*(T_wall-T_cell)/δ  (isothermal)
#                AbstractNeumann/AbstractPhysical → 0               (adiabatic)
#                fallback → κ*(∇T·n)
#   Slip/Symmetry           → nothing (zero shear, adiabatic)
#   PeriodicParent/Periodic → two-sided face-averaged gradients
#   fallback                → cell-centred gradient + bc_T heat flux
#
# bc_p: ghost pressure for Riemann paths only (irrelevant at Wall/Slip/Symmetry).
# bc_T: ghost temperature for Riemann paths; heat-flux selector at Wall.
# bc_nut: not used by flux chains; affects mueff via turbulence! → update_nueff!.
#
# ============================================================
# Flux scheme selector types
# ============================================================

"""Rusanov (Local Lax-Friedrichs) flux scheme."""
struct Rusanov end

"""HLLC (Harten-Lax-van Leer Contact) flux scheme."""
struct HLLC end

# ============================================================
# Spatial reconstruction types
# ============================================================

"""Van Leer smooth limiter: ψ(r) = (r + |r|) / (1 + |r|)"""
struct VanLeer end

"""MinMod limiter (most diffusive TVD): ψ(r) = max(0, min(1, r))"""
struct MinMod end

"""Superbee limiter (least diffusive TVD): ψ(r) = max(0, min(2r, 1), min(r, 2))"""
struct Superbee end

"""
    MUSCL{L}()

2nd-order MUSCL reconstruction with slope limiter `L` (VanLeer, MinMod, or Superbee).
Set via `reconstruction = MUSCL{VanLeer}()` in the schemes NamedTuple.
"""
struct MUSCL{L}
    MUSCL{L}() where L = new{L}()
end

# ============================================================
# Time stepping selector types
# ============================================================

"""Forward Euler (1st order) explicit time stepping."""
struct FEuler end

"""
SSP-RK2 / Heun's method (2nd order) explicit time stepping.
  W^(1)   = W^n   - (dt/V)*R(W^n)
  W^(2)   = W^(1) - (dt/V)*R(W^(1))
  W^{n+1} = 0.5*(W^n + W^(2))
"""
struct RK2 end

# ============================================================
# Limiter functions
# ============================================================

@inline limiter_value(::VanLeer,  r::T) where T = (r + abs(r)) / (one(T) + abs(r))
@inline limiter_value(::MinMod,   r::T) where T = max(zero(T), min(one(T), r))
@inline limiter_value(::Superbee, r::T) where T = max(zero(T), min(2*r, one(T)), min(r, 2*one(T)))

# ============================================================
# MUSCL reconstruction — scalar and vector helpers
# ============================================================

# Scalar MUSCL reconstruction using pre-computed gradient projections onto dLR = delta*e.
# Returns reconstructed face values (left, right) with TVD slope limiter.
@inline function _muscl_scalar(lim, φL::TF, φR::TF, projL::TF, projR::TF) where TF
    Δ = φR - φL
    abs(Δ) < TF(1e-30) && return φL, φR
    rL   = TF(2) * projL / Δ - one(TF)
    rR   = TF(2) * projR / Δ - one(TF)
    half = TF(0.5)
    return φL + half * limiter_value(lim, rL) * Δ,
           φR - half * limiter_value(lim, rR) * Δ
end

# ── Reconstruct dispatch ─────────────────────────────────────────────────────

"""1st-order reconstruction — returns cell-centred values unchanged."""
@inline function reconstruct(
    ::Upwind,
    rhoL, rhoR, UL, UR, pL, pR, args...
)
    return rhoL, rhoR, UL, UR, pL, pR
end

"""
2nd-order MUSCL reconstruction with slope limiter `L`.
Slope ratio (Darwish & Moukalled): r = 2*(∇φ·dLR)/Δ − 1.
Velocity components reconstructed via gradU * dLR (matrix–vector product).
ρ and p clamped to positive floor to prevent NaN in sqrt(γp/ρ).
"""
@inline function reconstruct(
    ::MUSCL{L},
    rhoL::TF, rhoR::TF,
    UL::SV,   UR::SV,
    pL::TF,   pR::TF,
    gradRhoL, gradRhoR,
    gradUL,   gradUR,
    gradPL,   gradPR,
    dLR
) where {TF, SV<:SVector{3,TF}, L}
    lim = L()

    # Density
    rhoLf, rhoRf = _muscl_scalar(lim, rhoL, rhoR, gradRhoL ⋅ dLR, gradRhoR ⋅ dLR)

    # Pressure
    pLf, pRf = _muscl_scalar(lim, pL, pR, gradPL ⋅ dLR, gradPR ⋅ dLR)

    # Velocity: gradU is SMatrix{3,3} (M[i,j]=∂U_i/∂x_j); M*dLR projects each component
    projUL = gradUL * dLR
    projUR = gradUR * dLR
    UxLf, UxRf = _muscl_scalar(lim, UL[1], UR[1], projUL[1], projUR[1])
    UyLf, UyRf = _muscl_scalar(lim, UL[2], UR[2], projUL[2], projUR[2])
    UzLf, UzRf = _muscl_scalar(lim, UL[3], UR[3], projUL[3], projUR[3])

    floor_val = TF(1e-10)
    return max(rhoLf, floor_val), max(rhoRf, floor_val),
           SVector{3,TF}(UxLf, UyLf, UzLf), SVector{3,TF}(UxRf, UyRf, UzRf),
           max(pLf, floor_val), max(pRf, floor_val)
end

# ============================================================
# Workspace struct
# ============================================================

struct GodunovWorkspace{SF<:ScalarField, VF<:VectorField, V<:AbstractVector}
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

@inline function ghost_velocity(bc::Wall, U_int::SV, n::SV) where {TF, SV<:SVector{3,TF}}
    U_wall = SV(TF(bc.value[1]), TF(bc.value[2]), TF(bc.value[3]))
    2*U_wall - U_int
end

@inline function ghost_velocity(bc::Symmetry, U_int::SV, n::SV) where {TF, SV<:SVector{3,TF}}
    # Reflect normal component
    U_int - 2*(U_int ⋅ n)*n
end

@inline function ghost_velocity(bc::Slip, U_int::SV, n::SV) where {TF, SV<:SVector{3,TF}}
    # Reflect normal component
    U_int - 2*(U_int ⋅ n)*n
end

@inline function ghost_velocity(bc::AbstractDirichlet, U_int::SV, n::SV) where {TF, SV<:SVector{3,TF}}
    U_bc = SV(TF(bc.value[1]), TF(bc.value[2]), TF(bc.value[3]))
    2*U_bc - U_int
end

@inline ghost_velocity(bc::AbstractNeumann, U_int::SV, n::SV) where {TF, SV<:SVector{3,TF}} = U_int
@inline ghost_velocity(bc::AbstractBoundary, U_int::SV, n::SV) where {TF, SV<:SVector{3,TF}} = U_int

# Extended versions with face/time/index context; fallbacks delegate to 3-arg versions.
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

@inline ghost_temperature(bc, face, T_int::TF, time, i) where TF = ghost_temperature(bc, T_int)

@inline function ghost_temperature(bc::DirichletFunction, face, T_int::TF, time, i) where TF
    TF(bc.value(face.centre, TF(time), i))
end

# --- Pressure ghost state ---

@inline ghost_pressure(bc::Dirichlet, p_int::TF, n) where TF = 2*TF(bc.value) - p_int
@inline ghost_pressure(bc::AbstractNeumann, p_int::TF, n) where TF = p_int
@inline ghost_pressure(bc::AbstractPhysicalConstraint, p_int::TF, n) where TF = p_int
@inline ghost_pressure(bc::AbstractBoundary, p_int::TF, n) where TF = p_int

# --- Temperature ghost state ---
# FixedTemperature/Dirichlet: ghost = T_bc (not 2*T_bc-T_int) for stability at high Mach,
# where stagnation T_int >> T_bc near the stagnation point.
@inline ghost_temperature(bc::FixedTemperature, T_int::TF) where TF = TF(bc.value.T)
@inline ghost_temperature(bc::Dirichlet, T_int::TF) where TF = TF(bc.value)

@inline ghost_temperature(bc::AbstractBoundary, T_int::TF) where TF = T_int

# ============================================================
# Rusanov (Local Lax-Friedrichs) flux
# ============================================================

@inline function rusanov_flux(
    UL::SV, UR::SV,
    pL::TF, pR::TF,
    rhoL::TF, rhoR::TF,
    normal::SV, area::TF, gamma::TF
) where {TF, SV<:SVector{3,TF}}
    unL = UL ⋅ normal
    unR = UR ⋅ normal

    aL = sqrt(gamma * pL / rhoL)
    aR = sqrt(gamma * pR / rhoR)

    # Rusanov dissipation coefficient λ = max(|u|+a)
    lambda = max(abs(unL) + aL, abs(unR) + aR)

    # ρE = p/(γ-1) + 0.5*ρ|U|²;  H = (ρE+p)/ρ
    gm1 = gamma - one(TF)
    rhoEL = pL/gm1 + TF(0.5)*rhoL*(UL ⋅ UL)
    rhoER = pR/gm1 + TF(0.5)*rhoR*(UR ⋅ UR)
    HL = (rhoEL + pL) / rhoL
    HR = (rhoER + pR) / rhoR

    rhoUL = rhoL * UL
    rhoUR = rhoR * UR

    # Physical fluxes F(W)·n
    FL_rho  = rhoL*unL
    FR_rho  = rhoR*unR
    FL_rhoU = rhoUL*unL + pL*normal
    FR_rhoU = rhoUR*unR + pR*normal
    FL_rhoE = rhoL*HL*unL
    FR_rhoE = rhoR*HR*unR

    # F_Rusanov = 0.5*(F_L+F_R) - 0.5*λ*(W_R-W_L)
    half   = TF(0.5)
    F_rho  = area*(half*(FL_rho  + FR_rho)  - half*lambda*(rhoR  - rhoL))
    F_rhoU = area*(half*(FL_rhoU + FR_rhoU) - half*lambda*(rhoUR - rhoUL))
    F_rhoE = area*(half*(FL_rhoE + FR_rhoE) - half*lambda*(rhoER - rhoEL))

    return F_rho, F_rhoU, F_rhoE
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

    unL = UL ⋅ normal
    unR = UR ⋅ normal

    aL = sqrt(gamma * pL / rhoL)
    aR = sqrt(gamma * pR / rhoR)

    # ρE = p/(γ-1) + 0.5*ρ|U|²
    rhoEL = pL/gm1 + TF(0.5)*rhoL*(UL ⋅ UL)
    rhoER = pR/gm1 + TF(0.5)*rhoR*(UR ⋅ UR)

    # Wave speed estimates (Einfeldt bounds)
    SL = min(unL - aL, unR - aR)
    SR = max(unL + aL, unR + aR)

    # Contact wave speed S* (Rankine-Hugoniot)
    Sstar = (pR - pL + rhoL*unL*(SL - unL) - rhoR*unR*(SR - unR)) /
            (rhoL*(SL - unL) - rhoR*(SR - unR))

    # Physical fluxes F(W)·n
    HL = (rhoEL + pL) / rhoL
    HR = (rhoER + pR) / rhoR

    FL_rho  = rhoL*unL
    FL_rhoU = rhoL*unL*UL + pL*normal
    FL_rhoE = rhoL*HL*unL

    FR_rho  = rhoR*unR
    FR_rhoU = rhoR*unR*UR + pR*normal
    FR_rhoE = rhoR*HR*unR

    # HLLC star states (contact-preserving)
    coefL   = rhoL * (SL - unL) / (SL - Sstar)
    WL_rho  = coefL
    WL_rhoU = coefL * (UL + (Sstar - unL)*normal)
    WL_rhoE = coefL * (rhoEL/rhoL + (Sstar - unL)*(Sstar + pL/(rhoL*(SL - unL))))

    coefR   = rhoR * (SR - unR) / (SR - Sstar)
    WR_rho  = coefR
    WR_rhoU = coefR * (UR + (Sstar - unR)*normal)
    WR_rhoE = coefR * (rhoER/rhoR + (Sstar - unR)*(Sstar + pR/(rhoR*(SR - unR))))

    # Flux region selection (Toro §10.4)
    if SL >= zero(TF)           # supersonic left
        F_rho  = area*FL_rho;  F_rhoU = area*FL_rhoU;  F_rhoE = area*FL_rhoE
    elseif Sstar >= zero(TF)    # left star region
        F_rho  = area*(FL_rho  + SL*(WL_rho  - rhoL))
        F_rhoU = area*(FL_rhoU + SL*(WL_rhoU - rhoL*UL))
        F_rhoE = area*(FL_rhoE + SL*(WL_rhoE - rhoEL))
    elseif SR >= zero(TF)       # right star region
        F_rho  = area*(FR_rho  + SR*(WR_rho  - rhoR))
        F_rhoU = area*(FR_rhoU + SR*(WR_rhoU - rhoR*UR))
        F_rhoE = area*(FR_rhoE + SR*(WR_rhoE - rhoER))
    else                        # supersonic right
        F_rho  = area*FR_rho;  F_rhoU = area*FR_rhoU;  F_rhoE = area*FR_rhoE
    end

    return F_rho, F_rhoU, F_rhoE
end

@inline compute_inviscid_flux(::Rusanov, args...) = rusanov_flux(args...)
@inline compute_inviscid_flux(::HLLC,    args...) = hllc_flux(args...)

# ============================================================
# Kernels
# ============================================================

@kernel function _prim_to_cons!(rhoU, rhoE, rho, U, p, fluid)
    i = @index(Global)

    @uniform gamma = fluid.gamma.values

    @inbounds begin
        rho_i = rho[i]
        Ui = U[i]
        pi = p[i]

        rhoUi = rho_i * Ui
        rhoU.x[i] = rhoUi[1]
        rhoU.y[i] = rhoUi[2]
        rhoU.z[i] = rhoUi[3]

        gm1 = gamma - one(gamma)
        KE = oftype(rho_i, 0.5) * (Ui ⋅ Ui)
        rhoE.values[i] = pi/gm1 + rho_i*KE
    end
end

@kernel function _init_rho!(rho, p, T, fluid)
    i = @index(Global)

    @uniform R_gas = fluid.R.values

    @inbounds begin
        rho.values[i] = p[i] / (R_gas * T[i])
    end
end

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

# Inviscid flux — internal faces, cell-based loop (no atomics)
@kernel function _inviscid_flux_internal!(
    flux_scheme, recon_scheme,
    res_rho, res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE,
    rho, U, p, gradRho, gradU_recon, gradP, mesh, fluid
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
        acc_rhoU = zero(SVector{3,TF})
        acc_rhoE = zero(TF)

        for fi ∈ faces_range
            fID = cell_faces[fi]

            nsign = TF(cell_nsign[fi])
            (; area, normal, e, delta, ownerCells) = faces[fID]

            cL = ownerCells[1]
            cR = ownerCells[2]

            # L=ownerCells[1], R=ownerCells[2] (consistent orientation)
            rhoL = rho[cL];  UL = U[cL];  pL = p[cL]
            rhoR = rho[cR];  UR = U[cR];  pR = p[cR]

            dLR = delta * e
            rhoL, rhoR, UL, UR, pL, pR = reconstruct(
                recon_scheme, rhoL, rhoR, UL, UR, pL, pR,
                gradRho[cL], gradRho[cR],
                gradU_recon[cL], gradU_recon[cR],
                gradP[cL], gradP[cR],
                dLR
            )

            F_rho, F_rhoU, F_rhoE = compute_inviscid_flux(
                flux_scheme, UL, UR, pL, pR, rhoL, rhoR, normal, area, gamma_tf
            )

            acc_rho  += nsign * F_rho
            acc_rhoU += nsign * F_rhoU
            acc_rhoE += nsign * F_rhoE
        end

        res_rho.values[i]   += acc_rho
        res_rhoUx.values[i] += acc_rhoU[1]
        res_rhoUy.values[i] += acc_rhoU[2]
        res_rhoUz.values[i] += acc_rhoU[3]
        res_rhoE.values[i]  += acc_rhoE
    end
end

# Inviscid flux — boundary faces, face-based loop (atomics)
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

# ── Impermeable wall BCs (Wall, Slip, Symmetry): exact Euler wall flux ──────────
# U·n=0 → F_mass=0, F_momentum=p*n*A, F_energy=0.
# Riemann solver with mirror ghost (U_R=-U_L) injects spurious tangential momentum
# ∝ a*ρ*|U_tang|, causing velocity blow-up at supersonic walls.

@inline function _apply_inviscid_bc!(
    flux_scheme, bc_U::Union{Wall, Slip, Symmetry}, bc_p, bc_T,
    res_rho, res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE,
    rho, U, p, T, mesh, fluid, fID, time
)
    (; faces) = mesh
    face = faces[fID]
    (; area, normal, ownerCells) = face
    cID = ownerCells[1]

    Fp = p[cID] * area * normal
    Atomix.@atomic res_rhoUx.values[cID] += Fp[1]
    Atomix.@atomic res_rhoUy.values[cID] += Fp[2]
    Atomix.@atomic res_rhoUz.values[cID] += Fp[3]
end

# ── Shared atomic accumulation helper for Riemann-path BC handlers ───────────────

@inline function _accumulate_inviscid!(
    res_rho, res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE,
    cID, F_rho, F_rhoU::SVector{3}, F_rhoE
)
    Atomix.@atomic res_rho.values[cID]   += F_rho
    Atomix.@atomic res_rhoUx.values[cID] += F_rhoU[1]
    Atomix.@atomic res_rhoUy.values[cID] += F_rhoU[2]
    Atomix.@atomic res_rhoUz.values[cID] += F_rhoU[3]
    Atomix.@atomic res_rhoE.values[cID]  += F_rhoE
end

# ── Outlet BC: zero-gradient outflow with reservoir-based backflow ───────────────
# Outflow (un_L > 0): UR=UL → with Zerogradient bc_p/bc_T: L=R → zero dissipation → exact flux.
# Backflow (un_L ≤ 0): UR=0 (stagnant ghost) at bc_p/bc_T thermodynamic state.
# Stagnant ghost avoids wall-flux pressure lock-up (p*n*A feedback loop).
@inline function _apply_inviscid_bc!(
    flux_scheme, bc_U::Outlet, bc_p, bc_T,
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

    UL   = U[cID]
    pL   = p[cID]
    TL   = T[cID]
    rhoL = rho[cID]

    un_L = UL ⋅ normal

    pR = ghost_pressure(bc_p, face, pL, normal, time, i)
    TR = ghost_temperature(bc_T, face, TL, time, i)
    pR   = max(pR,   TF(1e-10))
    TR   = max(TR,   TF(1e-10))
    rhoR = max(pR / (R_gas * TR), TF(1e-10))

    if un_L <= zero(TF)
        UR = zero(UL)   # backflow: stagnant exterior ghost
    else
        UR = UL         # outflow: zero-gradient (L=R → no dissipation)
    end

    F_rho, F_rhoU, F_rhoE = compute_inviscid_flux(
        flux_scheme, UL, UR, pL, pR, rhoL, rhoR, normal, area, gamma
    )

    _accumulate_inviscid!(res_rho, res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE,
                          cID, F_rho, F_rhoU, F_rhoE)
end

# ── AbstractDirichlet: ghost UR = 2*U_bc - UL (method of images) ─────────────────
# Face-average = U_bc exactly; Rusanov dissipation ∝ |UL-U_bc| damps reversed flow.
# If |un_bc| ≤ 1e-10 (tangential prescription): exact wall flux instead of Riemann solve.
@inline function _apply_inviscid_bc!(
    flux_scheme, bc_U::AbstractDirichlet, bc_p, bc_T,
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
    pL = p[cID]
    TL = T[cID]

    # Ghost: UR = 2*U_bc - UL; recover U_bc to check for tangential prescription
    UR_ghost = ghost_velocity(bc_U, face, UL, normal, time, i)
    U_bc  = TF(0.5) * (UL + UR_ghost)
    un_bc = U_bc ⋅ normal

    if abs(un_bc) <= TF(1e-10)
        # Prescribed velocity is tangential → impermeable exact wall flux.
        Fp = pL * area * normal
        Atomix.@atomic res_rhoUx.values[cID] += Fp[1]
        Atomix.@atomic res_rhoUy.values[cID] += Fp[2]
        Atomix.@atomic res_rhoUz.values[cID] += Fp[3]
        return
    end

    rhoL = rho[cID]
    pR   = ghost_pressure(bc_p, face, pL, normal, time, i)
    TR   = ghost_temperature(bc_T, face, TL, time, i)
    pR   = max(pR,   TF(1e-10))
    TR   = max(TR,   TF(1e-10))
    rhoR = max(pR / (R_gas * TR), TF(1e-10))

    F_rho, F_rhoU, F_rhoE = compute_inviscid_flux(
        flux_scheme, UL, UR_ghost, pL, pR, rhoL, rhoR, normal, area, gamma
    )

    _accumulate_inviscid!(res_rho, res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE,
                          cID, F_rho, F_rhoU, F_rhoE)
end

# ── Fallback: Neumann, Zerogradient, Extrapolated, far-field, etc. ──────────────
# Impermeability check: |un_face| ≤ 1e-10 → exact wall flux; otherwise Riemann solve.
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
    pL = p[cID]
    TL = T[cID]

    UR = ghost_velocity(bc_U, face, UL, normal, time, i)
    un_face = TF(0.5) * ((UL + UR) ⋅ normal)

    if abs(un_face) <= TF(1e-10)
        # Impermeable: exact wall flux
        Fp = pL * area * normal
        Atomix.@atomic res_rhoUx.values[cID] += Fp[1]
        Atomix.@atomic res_rhoUy.values[cID] += Fp[2]
        Atomix.@atomic res_rhoUz.values[cID] += Fp[3]
        return
    end

    rhoL = rho[cID]
    pR   = ghost_pressure(bc_p, face, pL, normal, time, i)
    TR   = ghost_temperature(bc_T, face, TL, time, i)
    pR   = max(pR,   TF(1e-10))
    TR   = max(TR,   TF(1e-10))
    rhoR = max(pR / (R_gas * TR), TF(1e-10))

    F_rho, F_rhoU, F_rhoE = compute_inviscid_flux(
        flux_scheme, UL, UR, pL, pR, rhoL, rhoR, normal, area, gamma
    )

    _accumulate_inviscid!(res_rho, res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE,
                          cID, F_rho, F_rhoU, F_rhoE)
end

# Periodic BC: Riemann solve with partner cell as right state.
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

    rhoL = rho[cID];  UL = U[cID];  pL = p[cID]
    rhoR = rho[pcID]; UR = U[pcID]; pR = p[pcID]

    F_rho, F_rhoU, F_rhoE = compute_inviscid_flux(
        flux_scheme, UL, UR, pL, pR, rhoL, rhoR, normal, area, gamma
    )

    _accumulate_inviscid!(res_rho, res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE,
                          cID, F_rho, F_rhoU, F_rhoE)
end

# Per-cell CFL time step: λ = |U| + a + ν_eff/dx → dt = CFL*dx/λ (combined convective+diffusive)
@kernel function _compute_dt_cell!(dt_cell, rho, U, p, cells, fluid, cfl, dim_exp, nu_eff)
    i = @index(Global)

    @uniform gamma = fluid.gamma.values

    @inbounds begin
        TF = eltype(rho.values)
        rho_i    = rho[i]
        Ui       = U[i]
        pi       = p[i]
        Vi       = cells[i].volume
        nu_eff_i = nu_eff[i]

        ai     = sqrt(TF(gamma) * pi / rho_i)
        Umag   = sqrt(Ui ⋅ Ui)
        dx     = Vi^dim_exp
        lambda = Umag + ai + nu_eff_i / (dx + TF(1e-30))

        dt_cell[i] = cfl * dx / (lambda + TF(1e-30))
    end
end

@kernel function _compute_nu_eff_cell!(nu_eff, nu_mol, nut)
    i = @index(Global)
    @inbounds nu_eff[i] = nu_mol + nut.values[i]
end

# Cell-level ν_eff for diffusive CFL; mirrors update_nueff! dispatch (Laminar has no nut field).
function update_nu_eff_cell!(nu_eff, nu_mol, turb_model, backend, workgroup, n_cells)
    if typeof(turb_model) <: Laminar
        @. nu_eff = nu_mol
    else
        kernel! = _compute_nu_eff_cell!(_setup(backend, workgroup, n_cells)...)
        kernel!(nu_eff, nu_mol, turb_model.nut)
    end
end

"""Return fixed dt from `runtime.dt[1]` (no kernel launch)."""
function compute_dt!(workspace, model, runtime::Runtime{<:Any,<:Any,<:Any,Nothing}, config, dim_exp)
    TF = eltype(workspace.dt_cell)
    return TF(runtime.dt[1])
end

"""Compute CFL-limited dt via per-cell kernel; update `runtime.dt` in place."""
function compute_dt!(workspace, model, runtime::Runtime{<:Any,<:Any,<:Any,<:AdaptiveTimeStepping}, config, dim_exp)
    TF = eltype(workspace.dt_cell)
    cfl = TF(runtime.adaptive.maxCo)
    (; dt_cell, nu_eff) = workspace
    (; rho) = model.fluid
    (; U, p) = model.momentum
    mesh = model.domain
    (; backend, workgroup) = config.hardware
    n_cells = length(mesh.cells)
    kernel! = _compute_dt_cell!(_setup(backend, workgroup, n_cells)...)
    kernel!(dt_cell, rho, U, p, mesh.cells, model.fluid, cfl, dim_exp, nu_eff)
    dt = minimum(dt_cell)
    runtime.dt .= dt
    return dt
end

@kernel function _forward_euler!(rho, rhoU, rhoE, res_rho, res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE, cells, dt)
    i = @index(Global)

    @inbounds begin
        TF = eltype(rho.values)
        V  = cells[i].volume
        factor = dt / V

        rho.values[i]   -= factor * res_rho.values[i]
        rhoU.x[i]       -= factor * res_rhoUx.values[i]
        rhoU.y[i]       -= factor * res_rhoUy.values[i]
        rhoU.z[i]       -= factor * res_rhoUz.values[i]
        rhoE.values[i]  -= factor * res_rhoE.values[i]

        rho.values[i] = max(rho.values[i], TF(1e-10))
    end
end

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

        rho_i  = rho[i]
        rhoE_i = rhoE[i]

        Ui = rhoU[i] / rho_i
        U.x[i] = Ui[1];  U.y[i] = Ui[2];  U.z[i] = Ui[3]

        KE  = TF(0.5) * (Ui ⋅ Ui)
        p_i = max((gamma_tf - one(TF)) * (rhoE_i - rho_i*KE), TF(1e-10))   # p = (γ-1)*(ρE-ρKE)
        p.values[i] = p_i

        T_i = max(p_i / (rho_i * R_gas_tf), TF(1e-10))                      # T = p/(ρR)
        T.values[i] = T_i

        a = sqrt(gamma_tf * p_i / rho_i)
        Mach.values[i] = sqrt(Ui ⋅ Ui) / (a + TF(1e-30))
    end
end

# ============================================================
# Viscous flux kernels
# ============================================================

# Viscous flux — internal faces, cell-based loop (no atomics)
@kernel function _viscous_flux_internal!(
    res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE,
    U, T, gradU, gradT, mueff, kappa_eff, mesh
)
    i = @index(Global)

    @uniform begin
        (; cells, faces, cell_faces, cell_nsign) = mesh
    end

    @inbounds begin
        (; faces_range) = cells[i]

        TF = eltype(res_rhoUx.values)
        two_thirds = TF(2)/TF(3)

        acc_rhoU = zero(SVector{3,TF})
        acc_rhoE = zero(TF)

        for fi ∈ faces_range
            fID     = cell_faces[fi]
            nsign   = TF(cell_nsign[fi])
            (; area, normal, e, delta, ownerCells) = faces[fID]

            cL = ownerCells[1]
            cR = ownerCells[2]

            gradU_f = TF(0.5) * (gradU[cL] + gradU[cR])
            gradT_f = TF(0.5) * (gradT[cL] + gradT[cR])

            # Non-orthogonal correction: ∇φ·n ≈ (φ_R-φ_L)/δ*(e·n) + ∇φ_f·(n-(e·n)*e)
            en = e ⋅ normal

            dU_over_delta = (U[cR] - U[cL]) / delta
            gradU_n  = dU_over_delta * en + gradU_f * (normal - en * e)
            gradUt_n = gradU_f' * normal

            divU    = gradU_f[1,1] + gradU_f[2,2] + gradU_f[3,3]
            mueff_f = mueff[fID]
            tau_n   = mueff_f * (gradU_n + gradUt_n - two_thirds * divU * normal)

            U_f = TF(0.5) * (U[cL] + U[cR])

            dT_over_delta = (T[cR] - T[cL]) / delta
            gradT_n  = dT_over_delta * en + gradT_f ⋅ (normal - en * e)

            kf       = kappa_eff[fID]
            F_visc_E = (U_f ⋅ tau_n) + kf * gradT_n

            acc_rhoU -= nsign * tau_n * area
            acc_rhoE -= nsign * F_visc_E * area
        end

        res_rhoUx.values[i] += acc_rhoU[1]
        res_rhoUy.values[i] += acc_rhoU[2]
        res_rhoUz.values[i] += acc_rhoU[3]
        res_rhoE.values[i]  += acc_rhoE
    end
end

# ============================================================
# Wall heat flux helpers — dispatch on energy BC type
# ============================================================

@inline wall_heat_flux(::AbstractNeumann, kf, gradT, T, normal, delta, cID, fID) = zero(kf)
@inline wall_heat_flux(::AbstractPhysicalConstraint, kf, gradT, T, normal, delta, cID, fID) = zero(kf)

# Isothermal (FixedTemperature): κ*(T_wall-T_cell)/δ
@inline function wall_heat_flux(
    bc_T::FixedTemperature, kf, gradT, T, normal, delta, cID, fID
)
    T_wall = typeof(kf)(bc_T.value.T)
    kf * (T_wall - T[cID]) / delta
end

# Isothermal (Dirichlet on T): κ*(T_wall-T_cell)/δ
@inline function wall_heat_flux(
    bc_T::Dirichlet, kf, gradT, T, normal, delta, cID, fID
)
    T_wall = typeof(kf)(bc_T.value)
    kf * (T_wall - T[cID]) / delta
end

# Fallback: κ*(∇T·n) from cell-centred gradient
@inline function wall_heat_flux(
    bc_T, kf, gradT, T, normal, delta, cID, fID
)
    kf * (gradT[cID] ⋅ normal)
end

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

# Generic viscous BC: cell-centred gradient (two-point 1/δ correction removed — amplifies
# stresses at thin cells). Heat flux dispatched on bc_T.
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

    Uf_face = Uf[fID]
    gradU_f = gradU[cID]

    divU    = gradU_f[1,1] + gradU_f[2,2] + gradU_f[3,3]
    mueff_f = mueff[fID]
    tau_n   = mueff_f * ((gradU_f + gradU_f') * normal - two_thirds * divU * normal)

    kf       = kappa_eff[fID]
    q_wall   = wall_heat_flux(bc_T, kf, gradT, T, normal, delta, cID, fID)
    F_visc_E = (Uf_face ⋅ tau_n) + q_wall

    tau_n_area = tau_n * area
    Atomix.@atomic res_rhoUx.values[cID] -= tau_n_area[1]
    Atomix.@atomic res_rhoUy.values[cID] -= tau_n_area[2]
    Atomix.@atomic res_rhoUz.values[cID] -= tau_n_area[3]
    Atomix.@atomic res_rhoE.values[cID]  -= F_visc_E * area
end

# Wall (no-slip): two-point gradient (U_wall-U_cell)/δ⊗n — orthogonal, avoids thin-cell amplification.
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

    U_wall  = SVector{3,TF}(TF(bc_U.value[1]), TF(bc_U.value[2]), TF(bc_U.value[3]))
    U_cell  = U[cID]

    dU      = (U_wall - U_cell) / delta
    gradU_f = dU * normal'   # outer product → ∇U_ij ≈ dU_i * n_j
    divU    = gradU_f[1,1] + gradU_f[2,2] + gradU_f[3,3]
    mueff_f = mueff[fID]
    tau_n   = mueff_f * ((gradU_f + gradU_f') * normal - two_thirds * divU * normal)

    kf       = kappa_eff[fID]
    q_wall   = wall_heat_flux(bc_T, kf, gradT, T, normal, delta, cID, fID)
    F_visc_E = (U_wall ⋅ tau_n) + q_wall

    tau_n_area = tau_n * area
    Atomix.@atomic res_rhoUx.values[cID] -= tau_n_area[1]
    Atomix.@atomic res_rhoUy.values[cID] -= tau_n_area[2]
    Atomix.@atomic res_rhoUz.values[cID] -= tau_n_area[3]
    Atomix.@atomic res_rhoE.values[cID]  -= F_visc_E * area
end

# Slip/Symmetry: zero shear + adiabatic → no viscous flux.
@inline function _apply_viscous_bc!(
    bc_U::Union{Slip, Symmetry}, bc_T,
    res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE,
    U, Uf, gradU, gradT, T, mueff, kappa_eff, mesh, fID
)
    nothing
end

# Periodic: two-sided face-averaged gradients (consistent with internal faces).
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

    gradU_f = TF(0.5) * (gradU[cID] + gradU[pcID])
    gradT_f = TF(0.5) * (gradT[cID] + gradT[pcID])

    divU    = gradU_f[1,1] + gradU_f[2,2] + gradU_f[3,3]
    mueff_f = mueff[fID]
    tau_n   = mueff_f * ((gradU_f + gradU_f') * normal - two_thirds * divU * normal)

    Uf_face  = TF(0.5) * (U[cID] + U[pcID])
    kf       = kappa_eff[fID]
    F_visc_E = (Uf_face ⋅ tau_n) + kf * (gradT_f ⋅ normal)

    tau_n_area = tau_n * area
    Atomix.@atomic res_rhoUx.values[cID] -= tau_n_area[1]
    Atomix.@atomic res_rhoUy.values[cID] -= tau_n_area[2]
    Atomix.@atomic res_rhoUz.values[cID] -= tau_n_area[3]
    Atomix.@atomic res_rhoE.values[cID]  -= F_visc_E * area
end

# ============================================================
# Setup and main solver loop
# ============================================================

"""
    godunov!(model, config; output=VTK())

Explicit Godunov-type compressible solver on conservative variables [ρ, ρU, ρE].
Dispatched from `run!` when `model.fluid isa SupersonicFlow`.
"""
function godunov!(model, config; output=VTK())
    residuals = _setup_godunov(model, config; output=output)
    return residuals
end

function update_recon_gradients!(recon_scheme, gradRho, gradP, rhof, pf, rho, p, boundaries, time, config)
    recon_scheme isa Upwind && return
    grad!(gradRho, rhof, rho, time, config)
    grad!(gradP, pf, p, boundaries.p, time, config)
end

function update_thermo_coeffs!(mueff, kappa_eff, rhof, nueff, cp_val, Pr_val)
    @. mueff.values     = rhof.values * nueff.values
    @. kappa_eff.values = mueff.values * cp_val / Pr_val
end

function _setup_godunov(model, config; output=VTK())
    (; U, p, Uf, pf) = model.momentum
    (; rho, nu) = model.fluid
    mesh = model.domain
    (; hardware, runtime, schemes, boundaries) = config
    (; backend, workgroup) = hardware

    @info "Allocating GodunovWorkspace..."

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

    workspace = GodunovWorkspace(
        rhoU, rhoE, res_rho, res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE, Mach, dt_cell,
        nu_eff_cell, rho_0, rhoUx_0, rhoUy_0, rhoUz_0, rhoE_0
    )

    @info "Allocating viscous and turbulence fields..."

    gradU  = Grad{schemes.U.gradient}(U)
    gradUT = T(gradU)
    Uf_s   = FaceVectorField(mesh)  # separate Uf for StrainRate
    S      = StrainRate(gradU, gradUT, U, Uf_s)

    T_field = model.energy.T
    gradT   = Grad{schemes.T.gradient}(T_field)

    # MUSCL reconstruction gradients (gradU reused from StrainRate)
    gradRho = Grad{schemes.p.gradient}(rho)
    gradP   = Grad{schemes.p.gradient}(p)

    nueff     = FaceScalarField(mesh)
    mueff     = FaceScalarField(mesh)
    kappa_eff = FaceScalarField(mesh)
    mdotf     = FaceScalarField(mesh)
    prev      = KernelAbstractions.zeros(backend, TF, n_cells)

    @info "Initialising turbulence model..."
    turbulenceModel, config = initialise(model.turbulence, model, mdotf, nothing, config)

    @info "Initialising density from p and T..."

    ndrange = n_cells
    kernel! = _init_rho!(_setup(backend, workgroup, ndrange)...)
    kernel!(rho, p, T_field, model.fluid)

    residuals = GODUNOV(
        model, workspace, turbulenceModel,
        S, gradT, gradRho, gradP, nueff, mueff, kappa_eff, mdotf, prev,
        config; output=output
    )
    return residuals
end

function compute_residuals!(
    workspace, flux_scheme, recon_scheme, boundaries, model,
    gradU, gradRho, gradP, gradT, mueff, kappa_eff,
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

    kernel! = _zero_residuals!(_setup(backend, workgroup, n_cells)...)
    kernel!(res_rho, res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE)

    kernel! = _inviscid_flux_internal!(_setup(backend, workgroup, n_cells)...)
    kernel!(flux_scheme, recon_scheme, res_rho, res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE,
            rho, U, p, gradRho, gradU, gradP, mesh, model.fluid)

    kernel! = _viscous_flux_internal!(_setup(backend, workgroup, n_cells)...)
    kernel!(res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE, U, T, gradU, gradT, mueff, kappa_eff, mesh)

    kernel! = _inviscid_bc_flux!(_setup(backend, workgroup, n_bfaces)...)
    kernel!(flux_scheme, boundaries.U, boundaries.p, boundaries.T,
            res_rho, res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE,
            rho, U, p, T, mesh, model.fluid, time)

    kernel! = _viscous_bc_flux!(_setup(backend, workgroup, n_bfaces)...)
    kernel!(boundaries.U, boundaries.T, res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE,
            U, Uf, gradU, gradT, T, mueff, kappa_eff, mesh)
end

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

"""Forward-Euler step: W^{n+1} = W^n - (dt/V)*R(W^n). Call recover_primitives! after."""
function step!(
    ::FEuler, workspace, model, boundaries, flux_scheme, recon_scheme,
    gradU, gradRho, gradP, gradT, nueff, mueff, kappa_eff, rhof, mdotf, dt, time, config
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
SSP-RK2 step: W^(1)=W^n-(dt/V)*R(W^n), W^(2)=W^(1)-(dt/V)*R(W^(1)), W^{n+1}=0.5*(W^n+W^(2)).
R(W^n) must be pre-computed. Call recover_primitives! after.
"""
function step!(
    ::RK2, workspace, model, boundaries, flux_scheme, recon_scheme,
    gradU, gradRho, gradP, gradT, nueff, mueff, kappa_eff, rhof, mdotf, dt, time, config
)
    (; rhoU, rhoE, res_rho, res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE) = workspace
    (; rho_0, rhoUx_0, rhoUy_0, rhoUz_0, rhoE_0) = workspace
    (; rho) = model.fluid
    (; U, p, Uf, pf) = model.momentum
    (; T, Tf) = model.energy
    (; hardware, schemes) = config
    (; backend, workgroup) = hardware
    mesh = model.domain
    n_cells = length(mesh.cells)
    TF = _get_float(mesh)
    cp_val = TF(model.fluid.cp.values)
    Pr_val = TF(model.fluid.Pr.values)

    # Stage 1: save W^n, Euler update using R(W^n)
    kernel! = _save_conservative!(_setup(backend, workgroup, n_cells)...)
    kernel!(rho_0, rhoUx_0, rhoUy_0, rhoUz_0, rhoE_0, rho, rhoU, rhoE)

    kernel! = _forward_euler!(_setup(backend, workgroup, n_cells)...)
    kernel!(rho, rhoU, rhoE, res_rho, res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE, mesh.cells, dt)

    # Update primitives and face fields at W^(1) for stage-2 residual
    recover_primitives!(workspace, model, config)
    interpolate_primitive_fields!(model, boundaries, rhof, mdotf, time, config)
    grad!(gradT, Tf, T, boundaries.T, time, config)
    limit_gradient!(schemes.T.limiter, gradT, T, config)
    grad!(gradU, Uf, U, boundaries.U, time, config)
    limit_gradient!(schemes.U.limiter, gradU, U, config)
    update_recon_gradients!(recon_scheme, gradRho, gradP, rhof, pf, rho, p, boundaries, time, config)
    update_thermo_coeffs!(mueff, kappa_eff, rhof, nueff, cp_val, Pr_val)

    # Stage 2: R(W^(1)), Euler update → W^(2)
    compute_residuals!(workspace, flux_scheme, recon_scheme, boundaries, model,
                       gradU, gradRho, gradP, gradT, mueff, kappa_eff, Uf, mesh, time, config)

    kernel! = _forward_euler!(_setup(backend, workgroup, n_cells)...)
    kernel!(rho, rhoU, rhoE, res_rho, res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE, mesh.cells, dt)

    # Convex average: W^{n+1} = 0.5*(W^n + W^(2))
    kernel! = _rk2_average!(_setup(backend, workgroup, n_cells)...)
    kernel!(rho, rhoU, rhoE, rho_0, rhoUx_0, rhoUy_0, rhoUz_0, rhoE_0)
end

# ============================================================
# Main solver loop
# ============================================================

function GODUNOV(
    model, workspace, turbulenceModel,
    S, gradT, gradRho, gradP, nueff, mueff, kappa_eff, mdotf, prev,
    config; output=VTK()
)
    (; U, p, Uf, pf) = model.momentum
    (; rho, nu, R) = model.fluid
    (; T, Tf) = model.energy
    mesh = model.domain
    (; schemes, runtime, hardware, boundaries) = config
    (; iterations, write_interval) = runtime
    (; backend, workgroup) = hardware

    flux_scheme  = get(schemes, :flux, Rusanov())
    time_scheme  = get(schemes, :time_stepping, FEuler())
    recon_scheme = schemes.reconstruction

    (; rhoU, rhoE, res_rho) = workspace

    n_cells = length(mesh.cells)
    TF = _get_float(mesh)

    dim_exp = typeof(mesh) <: Mesh2 ? TF(0.5) : TF(1/3)   # dx = V^dim_exp

    outputWriter = initialise_writer(output, model.domain)

    (; gradU) = S
    cp_val = TF(model.fluid.cp.values)
    Pr_val = TF(model.fluid.Pr.values)
    rhof   = model.fluid.rhof

    @info "Initialising conservative variables from primitive fields..."

    kernel! = _prim_to_cons!(_setup(backend, workgroup, n_cells)...)
    kernel!(rhoU, rhoE, rho, U, p, model.fluid)

    time = TF(0.0)

    nu_mol = TF(model.fluid.nu.values)

    @info "Initialising face fields, gradients and viscous coefficients..."

    @. rho.values = p.values / (R.values * T.values)
    interpolate_primitive_fields!(model, boundaries, rhof, mdotf, time, config)

    update_nueff!(nueff, nu, model.turbulence, config)
    update_thermo_coeffs!(mueff, kappa_eff, rhof, nueff, cp_val, Pr_val)

    grad!(gradU, Uf, U, boundaries.U, time, config)
    limit_gradient!(schemes.U.limiter, gradU, U, config)
    grad!(gradT, Tf, T, boundaries.T, time, config)
    limit_gradient!(schemes.T.limiter, gradT, T, config)
    update_recon_gradients!(recon_scheme, gradRho, gradP, rhof, pf, rho, p, boundaries, time, config)

    @. workspace.nu_eff = nu_mol
    R_rho = ones(TF, iterations)

    @info "Starting GODUNOV time loop ($(typeof(time_scheme)))..."

    progress = Progress(iterations; dt=1.0, showspeed=true)

    for iteration ∈ 1:iterations

        # 1. Compute R(W^n)
        compute_residuals!(workspace, flux_scheme, recon_scheme, boundaries, model,
                           gradU, gradRho, gradP, gradT, mueff, kappa_eff, Uf, mesh, time, config)

        # 2. L2 density residual (before conservative update)
        rho_res = norm(res_rho.values) / sqrt(TF(n_cells))
        R_rho[iteration] = rho_res

        # 3. Time step (adaptive CFL or fixed)
        dt = compute_dt!(workspace, model, runtime, config, dim_exp)
        time += dt

        # 4. Advance W^n → W^{n+1} (FEuler: single update; RK2: two-stage SSP)
        step!(time_scheme, workspace, model, boundaries, flux_scheme, recon_scheme,
              gradU, gradRho, gradP, gradT, nueff, mueff, kappa_eff, rhof, mdotf, dt, time, config)

        # 5. Recover primitives from W^{n+1}
        recover_primitives!(workspace, model, config)

        # 6. Face interpolation and mass flux
        interpolate_primitive_fields!(model, boundaries, rhof, mdotf, time, config)

        # 7. Gradients (gradU updated by turbulence! below)
        grad!(gradT, Tf, T, boundaries.T, time, config)
        limit_gradient!(schemes.T.limiter, gradT, T, config)
        update_recon_gradients!(recon_scheme, gradRho, gradP, rhof, pf, rho, p, boundaries, time, config)

        # 8. Turbulence update (recomputes gradU and S internally)
        turbulence!(turbulenceModel, model, S, prev, time, config)

        # 9. Effective viscosity and thermal conductivity
        update_nueff!(nueff, nu, model.turbulence, config)
        update_thermo_coeffs!(mueff, kappa_eff, rhof, nueff, cp_val, Pr_val)

        # 10. Cell-level ν_eff for next iteration's diffusive CFL
        update_nu_eff_cell!(workspace.nu_eff, nu_mol, model.turbulence, backend, workgroup, n_cells)

        ProgressMeter.next!(progress, showvalues = [
            (:time, time), (:iter, iteration), (:dt, dt), (:continuity_error, rho_res),
            turbulenceModel.state.residuals...
        ])

        if iteration % write_interval + signbit(write_interval) == 0
            save_output(model, outputWriter, iteration, time, config)
        end

    end # time loop

    return (rho=R_rho,)
end
