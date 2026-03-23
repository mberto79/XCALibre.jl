export density_based!, Rusanov, HLLC

# ============================================================
# Flux scheme selector types
# ============================================================

"""User-facing flux scheme type — selects the Rusanov (Local Lax-Friedrichs) flux."""
struct Rusanov end

"""User-facing flux scheme type — selects the HLLC (Harten-Lax-van Leer Contact) flux."""
struct HLLC end

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

# --- Temperature ghost state (from he BC) ---

@inline function ghost_temperature(
    bc::FixedTemperature, T_int::TF, cp::TF, Tref::TF
) where TF
    # Isothermal wall: set ghost = T_wall directly.
    # The 2*T_wall - T_int extrapolation is unstable at high Mach numbers
    # because stagnation T_int >> 2*T_wall near the stagnation point.
    # Using ghost = T_wall gives T_face = 0.5*(T_int+T_wall) which is
    # always positive and converges to T_wall as T_int → T_wall.
    TF(bc.value.T)
end

@inline function ghost_temperature(
    bc::Dirichlet, T_int::TF, cp::TF, Tref::TF
) where TF
    # Dirichlet on he: he = cp*(T-Tref) → T = he/cp + Tref
    TF(bc.value) / cp + Tref
end

@inline function ghost_temperature(
    bc::AbstractBoundary, T_int::TF, cp::TF, Tref::TF
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
    flux_scheme, U_BCs, p_BCs, he_BCs,
    res_rho, res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE,
    rho, U, p, T, mesh, fluid, Tref
)
    fID = @index(Global)

    @inbounds _inviscid_bc_dispatch!(
        flux_scheme, U_BCs, p_BCs, he_BCs,
        res_rho, res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE,
        rho, U, p, T, mesh, fluid, Tref, fID
    )
end

# Generated dispatch over boundary patches (compile-time loop unrolling)
@generated function _inviscid_bc_dispatch!(
    flux_scheme, U_BCs, p_BCs, he_BCs,
    res_rho, res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE,
    rho, U, p, T, mesh, fluid, Tref, fID
)
    n = length(U_BCs.parameters)
    exprs = []
    for i ∈ 1:n
        ex = quote
            bc_U_i  = U_BCs[$i]
            bc_p_i  = p_BCs[$i]
            bc_he_i = he_BCs[$i]
            (; start, stop) = bc_U_i.IDs_range
            if start <= fID <= stop
                _apply_inviscid_bc!(
                    flux_scheme, bc_U_i, bc_p_i, bc_he_i,
                    res_rho, res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE,
                    rho, U, p, T, mesh, fluid, Tref, fID
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

# Apply inviscid flux for a single boundary face, dispatching on BC types
@inline function _apply_inviscid_bc!(
    flux_scheme, bc_U, bc_p, bc_he,
    res_rho, res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE,
    rho, U, p, T, mesh, fluid, Tref, fID
)
    (; faces) = mesh
    face = faces[fID]
    (; area, normal, ownerCells) = face
    cID = ownerCells[1]   # interior cell

    TF = eltype(rho.values)
    gamma = TF(fluid.gamma.values)
    R_gas = TF(fluid.R.values)
    cp    = TF(fluid.cp.values)
    Tref_tf = TF(Tref)

    # Interior (left) state
    rhoL = TF(rho[cID])
    UL   = U[cID]
    pL   = TF(p[cID])
    TL   = TF(T[cID])

    # Ghost (right) state from boundary conditions
    UR = ghost_velocity(bc_U, UL, normal)
    pR = ghost_pressure(bc_p, pL, normal)
    TR = ghost_temperature(bc_he, TL, cp, Tref_tf)

    # Clamp ghost state to physical values
    pR   = max(pR,   TF(1e-10))
    TR   = max(TR,   TF(1e-10))
    rhoR = pR / (R_gas * TR)
    rhoR = max(rhoR, TF(1e-10))

    # Inviscid flux (boundary face: outward normal = face normal, no nsign correction)
    F_rho, F_rhoUx, F_rhoUy, F_rhoUz, F_rhoE = compute_inviscid_flux(
        flux_scheme, UL, UR, pL, pR, rhoL, rhoR, normal, area, gamma
    )

    Atomix.@atomic res_rho.values[cID]  += F_rho
    Atomix.@atomic res_rhoUx.values[cID] += F_rhoUx
    Atomix.@atomic res_rhoUy.values[cID] += F_rhoUy
    Atomix.@atomic res_rhoUz.values[cID] += F_rhoUz
    Atomix.@atomic res_rhoE.values[cID] += F_rhoE
end

# CFL-limited time step per cell (simple cell-size estimate)
@kernel function _compute_dt_cell!(dt_cell, rho, U, p, cells, fluid, cfl, dim_exp)
    i = @index(Global)

    @uniform gamma = fluid.gamma.values

    @inbounds begin
        TF = eltype(rho.values)
        rho_i = TF(rho[i])
        Ui    = U[i]
        pi    = TF(p[i])
        Vi    = TF(cells[i].volume)

        ai      = sqrt(TF(gamma) * pi / rho_i)
        Umag    = sqrt(Ui ⋅ Ui)
        lambda  = Umag + ai

        dx = Vi^TF(dim_exp)
        dt_cell[i] = TF(cfl) * dx / (lambda + TF(1e-30))
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

# Boundary faces — face-based loop with atomics, same pattern as _rusanov_bc_flux!
@kernel function _viscous_bc_flux!(
    res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE,
    U, gradU, gradT, mueff, kappa_eff, mesh
)
    fID = @index(Global)

    @inbounds begin
        (; faces) = mesh
        face = faces[fID]
        (; area, normal, ownerCells) = face
        cID = ownerCells[1]  # interior cell

        TF = eltype(res_rhoUx.values)
        two_thirds = TF(2)/TF(3)

        # One-sided gradients from interior cell
        gradU_f = gradU[cID]
        gradT_f = gradT[cID]

        # Velocity divergence
        divU = gradU_f[1,1] + gradU_f[2,2] + gradU_f[3,3]

        # Viscous stress projection
        mueff_f = TF(mueff[fID])
        tau_n   = mueff_f * ((gradU_f + gradU_f') * normal - two_thirds * divU * normal)

        # Interior velocity at boundary face
        Ui = U[cID]

        # Viscous energy flux
        kf       = TF(kappa_eff[fID])
        F_visc_E = (Ui ⋅ tau_n) + kf * (gradT_f ⋅ normal)

        # Subtract viscous contribution (boundary faces are outward from cID)
        Atomix.@atomic res_rhoUx.values[cID] -= tau_n[1] * area
        Atomix.@atomic res_rhoUy.values[cID] -= tau_n[2] * area
        Atomix.@atomic res_rhoUz.values[cID] -= tau_n[3] * area
        Atomix.@atomic res_rhoE.values[cID]  -= F_visc_E * area
    end
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
    dt_cell  = KernelAbstractions.zeros(backend, TF, n_cells)

    workspace = DensityBasedWorkspace(
        rhoU, rhoE, res_rho, res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE, Mach, dt_cell
    )

    @info "Allocating viscous and turbulence fields..."

    # Gradient objects: created before T is bound to temperature (T type still available)
    gradU  = Grad{schemes.U.gradient}(U)
    gradUT = T(gradU)
    Uf_s   = FaceVectorField(mesh)  # Uf for StrainRate (separate from momentum Uf)
    S      = StrainRate(gradU, gradUT, U, Uf_s)

    # Temperature gradient (T_field = model.energy.T, Tf = model.energy.Tf for face values)
    T_field = model.energy.T
    gradT   = Grad{schemes.he.gradient}(T_field)

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

function DENSITY_BASED(
    model, workspace, turbulenceModel,
    S, gradT, nueff, mueff, kappa_eff, mdotf, prev,
    config; output=VTK()
)
    (; U, p, Uf, pf) = model.momentum
    (; rho, nu) = model.fluid
    T  = model.energy.T
    Tf = model.energy.Tf   # face temperature (FaceScalarField)
    mesh = model.domain
    (; solvers, schemes, runtime, hardware, boundaries) = config
    (; iterations, write_interval) = runtime
    (; backend, workgroup) = hardware

    # Flux scheme: user sets schemes = (..., flux = HLLC()) or flux = Rusanov(); default Rusanov
    flux_scheme = get(schemes, :flux, Rusanov())

    (; rhoU, rhoE, res_rho, res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE, Mach, dt_cell) = workspace

    n_cells  = length(mesh.cells)
    n_bfaces = length(mesh.boundary_cellsID)
    TF = _get_float(mesh)

    # CFL number from adaptive time stepping or default
    cfl = if !isnothing(runtime.adaptive)
        TF(runtime.adaptive.maxCo)
    else
        TF(0.5)
    end

    # Cell size exponent: 1/2 for 2D, 1/3 for 3D
    dim_exp = typeof(mesh) <: Mesh2 ? TF(0.5) : TF(0.333333)

    # Reference temperature from energy model coefficients
    Tref = if hasproperty(model.energy.coeffs, :Tref)
        TF(model.energy.coeffs.Tref)
    else
        TF(0.0)
    end

    outputWriter = initialise_writer(output, model.domain)

    # Extract gradU from the StrainRate object (updated by turbulence! each iteration)
    (; gradU) = S

    # Fluid constants for thermal conductivity: κ_eff = μ_eff * cp / Pr
    cp_val = TF(model.fluid.cp.values)
    Pr_val = TF(model.fluid.Pr.values)
    rhof   = model.fluid.rhof  # face density (FaceScalarField in SupersonicFlow)

    @info "Initialising conservative variables..."

    ndrange = n_cells
    kernel! = _prim_to_cons!(_setup(backend, workgroup, ndrange)...)
    kernel!(rhoU, rhoE, rho, U, p, model.fluid)

    time = TF(0.0)

    # Initial face interpolations for viscous fields
    interpolate!(rhof, rho, config)
    interpolate!(Tf, T, config)
    update_nueff!(nueff, nu, model.turbulence, config)
    @. mueff.values     = rhof.values * nueff.values
    @. kappa_eff.values = mueff.values * cp_val / Pr_val
    flux!(mdotf, Uf, config)
    @. mdotf.values *= rhof.values

    # Pre-allocate residual storage
    R_rho = ones(TF, iterations)

    @info "Starting DENSITY_BASED time loop..."

    progress = Progress(iterations; dt=1.0, showspeed=true)

    for iteration ∈ 1:iterations

        # 0. Update face density and temperature, compute gradients and effective viscosity
        interpolate!(rhof, rho, config)
        interpolate!(Tf, T, config)

        # Velocity gradients updated inside turbulence! (Laminar+Compressible calls grad! + limit_gradient!)
        turbulence!(turbulenceModel, model, S, prev, time, config)

        # Temperature gradient: T is a derived field (from cons-to-prim).
        # Do NOT use boundaries.he — FixedTemperature would apply cp*(T-Tref) instead of T.
        # Tf is already interpolated above; this version of grad! skips boundary correction.
        grad!(gradT, Tf, T, time, config)
        limit_gradient!(schemes.he.limiter, gradT, T, config)

        # Effective viscosity: nueff (kinematic, on faces) → mueff = rhof * nueff
        update_nueff!(nueff, nu, model.turbulence, config)
        @. mueff.values     = rhof.values * nueff.values
        @. kappa_eff.values = mueff.values * cp_val / Pr_val

        # 1. Zero residuals
        kernel! = _zero_residuals!(_setup(backend, workgroup, n_cells)...)
        kernel!(res_rho, res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE)

        # 2. Inviscid flux — internal faces (cell loop, no atomics)
        kernel! = _inviscid_flux_internal!(_setup(backend, workgroup, n_cells)...)
        kernel!(
            flux_scheme,
            res_rho, res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE,
            rho, U, p, mesh, model.fluid
        )

        # 2a. Viscous flux — internal faces (subtract from residuals)
        kernel! = _viscous_flux_internal!(_setup(backend, workgroup, n_cells)...)
        kernel!(
            res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE,
            U, gradU, gradT, mueff, kappa_eff, mesh
        )

        # 3. Inviscid flux — boundary faces (face loop, with atomics)
        kernel! = _inviscid_bc_flux!(_setup(backend, workgroup, n_bfaces)...)
        kernel!(
            flux_scheme, boundaries.U, boundaries.p, boundaries.he,
            res_rho, res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE,
            rho, U, p, T, mesh, model.fluid, Tref
        )

        # 3a. Viscous flux — boundary faces (subtract from residuals)
        kernel! = _viscous_bc_flux!(_setup(backend, workgroup, n_bfaces)...)
        kernel!(
            res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE,
            U, gradU, gradT, mueff, kappa_eff, mesh
        )

        # 4. Compute density residual before update (L2 norm)
        rho_res = norm(res_rho.values) / sqrt(TF(n_cells))
        R_rho[iteration] = rho_res

        # 5. CFL-limited global dt
        kernel! = _compute_dt_cell!(_setup(backend, workgroup, n_cells)...)
        kernel!(dt_cell, rho, U, p, mesh.cells, model.fluid, cfl, dim_exp)
        dt = minimum(dt_cell)

        # Update runtime dt (for output routines that need it)
        runtime.dt .= dt
        time += dt

        # 6. Forward Euler update of conservative variables
        kernel! = _forward_euler!(_setup(backend, workgroup, n_cells)...)
        kernel!(rho, rhoU, rhoE, res_rho, res_rhoUx, res_rhoUy, res_rhoUz, res_rhoE, mesh.cells, dt)

        # 7. Conservative → primitive recovery
        kernel! = _cons_to_prim!(_setup(backend, workgroup, n_cells)...)
        kernel!(U, p, T, Mach, rho, rhoU, rhoE, mesh.cells, model.fluid)

        # 8. Update face interpolations
        interpolate!(Uf, U, config)
        correct_boundaries!(Uf, U, boundaries.U, time, config)
        interpolate!(pf, p, config)
        correct_boundaries!(pf, p, boundaries.p, time, config)

        # 8a. Update mass flux for turbulence transport equations
        flux!(mdotf, Uf, config)
        @. mdotf.values *= rhof.values

        # 9. Progress and convergence check
        ProgressMeter.next!(
            progress, showvalues = [
                (:iter, iteration),
                (:dt, dt),
                (:rho_residual, rho_res),
                turbulenceModel.state.residuals...
            ]
        )

        if rho_res <= solvers.rho.convergence
            progress.n = iteration
            finish!(progress)
            @info "Density-based solver converged in $iteration iterations!"
            if !signbit(write_interval)
                save_output(model, outputWriter, iteration, time, config)
            end
            break
        end

        # 10. Write output at specified interval
        if iteration % write_interval + signbit(write_interval) == 0
            save_output(model, outputWriter, iteration, time, config)
        end

    end # time loop

    return (rho=R_rho,)
end
