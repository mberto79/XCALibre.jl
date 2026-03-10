export KOmegaLKE

# Model type definition (hold fields)
"""
    KOmegaLKE <: AbstractTurbulenceModel

kOmega model containing all kOmega field parameters.

### Fields
- `k` -- Turbulent kinetic energy ScalarField.
- `omega` -- Specific dissipation rate ScalarField.
- `kl` -- ScalarField.
- `nut` -- Eddy viscosity ScalarField.
- `kf` -- Turbulent kinetic energy FaceScalarField.
- `omegaf` -- Specific dissipation rate FaceScalarField.
- `klf` -- FaceScalarField.
- `nutf` -- Eddy viscosity FaceScalarField.
- `coeffs` -- Model coefficients.
- `Tu` -- Freestream turbulence intensity for model.
- `y` -- Near-wall distance for model.

"""
struct KOmegaLKE{S1,S2,S3,S4,F1,F2,F3,F4,C1,C2,Y,BC} <: AbstractRANSModel 
    k::S1
    omega::S2
    kl::S3
    nut::S4
    kf::F1
    omegaf::F2
    klf::F3
    nutf::F4
    coeffs::C1
    Tu::C2
    y::Y
    wallBCs::BC
end 
Adapt.@adapt_structure KOmegaLKE

# Model type definition (hold equation definitions and internal data)
struct KOmegaLKEModel{
    T,E1,E2,E3,F1,F2,F3,S1,S2,S3,S4,S5,S6,S7,V1,V2,State}
    turbulence::T
    k_eqn::E1
    ω_eqn::E2
    kl_eqn::E3
    nueffkLS::F1
    nueffkS::F2
    nueffωS::F3
    nuL::S1
    nuts::S2
    Ω::S3
    γ::S4
    fv::S5
    normU::S6
    Reυ::S7
    ∇k::V1
    ∇ω::V2
    state::State
end 
Adapt.@adapt_structure KOmegaLKEModel

# Model API constructor
RANS{KOmegaLKE}(; Tu, walls) = begin
    args = (Tu=Tu, walls=walls)
    ARG = typeof(args)
    RANS{KOmegaLKE,ARG}(args)
end

# Functor as constructor (internally called by Physics API): Returns fields and user data
(rans::RANS{KOmegaLKE, ARG})(mesh) where ARG = begin
    k = ScalarField(mesh)
    omega = ScalarField(mesh)
    kl = ScalarField(mesh)
    nut = ScalarField(mesh)
    kf = FaceScalarField(mesh)
    omegaf = FaceScalarField(mesh)
    klf = FaceScalarField(mesh)
    nutf = FaceScalarField(mesh)
    coeffs = (
        C1 = 0.02974,
        C2 = 59.79,
        C3 = 1.191,
        C4 = 1.65*10^-13,
        Cμ = 0.09,
        Cω1 = 0.52,
        Cω2 = 0.0708,
        Ccrit = 76500,
        CSS = 1.45,
        Cv = 0.43,
        σk = 0.5,
        σd = 0.125,
        σkL = 0.0125,
        σω = 0.5
    )
    Tu = rans.args.Tu

    # Allocate wall distance "y" and setup boundary conditions
    y = ScalarField(mesh)
    wallBCs = rans.args.walls
    KOmegaLKE(k, omega, kl, nut, kf, omegaf, klf, nutf, coeffs, Tu, y, wallBCs)
end

# Model initialisation
"""
    initialise(turbulence::KOmegaLKE, model::Physics{T,F,SO,M,Tu,E,D,BI}, mdotf, peqn, config
    ) where {T,F,SO,M,Tu,E,D,BI}

Initialisation of turbulent transport equations.

### Input
- `turbulence` -- turbulence model.
- `model`  -- Physics model defined by user.
- `mdtof`  -- Face mass flow.
- `peqn`   -- Pressure equation.
- `config` -- Configuration structure defined by user with solvers, schemes, runtime and 
          hardware structures set.

### Output
- `KOmegaLKEModel(
        turbulence,
        k_eqn,
        ω_eqn,
        kl_eqn,
        nueffkLS,
        nueffkS,
        nueffωS,
        nuL,
        nuts,
        Ω,
        γ,
        fv,
        normU,
        Reυ,
        ∇k,
        ∇ω, 
        state
    )`  -- Turbulence model structure.

"""
function initialise(
    turbulence::KOmegaLKE, model::Physics{T,F,SO,M,Tu,E,D,BI}, mdotf, peqn, config
    ) where {T,F,SO,M,Tu,E,D,BI}

    @info "Initialising k-ω LKE model..."

    # unpack turbulent quantities and configuration
    (; k, omega, kl, kf, omegaf, klf, y, wallBCs) = model.turbulence
    (; solvers, schemes, runtime, boundaries) = config
    mesh = mdotf.mesh
    eqn = peqn.equation

    nueffkLS = ScalarField(mesh)
    nueffkS = ScalarField(mesh)
    nueffωS = ScalarField(mesh)
    nueffkL = FaceScalarField(mesh)
    nueffk = FaceScalarField(mesh)
    nueffω = FaceScalarField(mesh)
    DkLf = ScalarField(mesh)
    Dkf = ScalarField(mesh)
    Dωf = ScalarField(mesh)
    PkL = ScalarField(mesh)
    Pk = ScalarField(mesh)
    Pω = ScalarField(mesh)
    dkdomegadx = ScalarField(mesh)
    normU = ScalarField(mesh)
    Reυ = ScalarField(mesh)
    nuL = ScalarField(mesh)
    nuts = ScalarField(mesh)
    Ω = ScalarField(mesh)
    γ = ScalarField(mesh)
    fv = ScalarField(mesh)
    ∇k = Grad{schemes.k.gradient}(k)
    ∇ω = Grad{schemes.p.gradient}(omega)

    kl_eqn = (
            Time{schemes.kl.time}(kl)
            + Divergence{schemes.kl.divergence}(mdotf, kl) 
            - Laplacian{schemes.kl.laplacian}(nueffkL, kl) 
            + Si(DkLf,kl) # DkLf = 2*nu/y^2
            ==
            Source(PkL)
        ) → eqn

    k_eqn = (
            Time{schemes.k.time}(k)
            + Divergence{schemes.k.divergence}(mdotf, k) 
            - Laplacian{schemes.k.laplacian}(nueffk, k) 
            + Si(Dkf,k) # Dkf = β⁺*omega
            ==
            Source(Pk)
        ) → eqn
    
    ω_eqn = (
            Time{schemes.omega.time}(omega)
            + Divergence{schemes.omega.divergence}(mdotf, omega) 
            - Laplacian{schemes.omega.laplacian}(nueffω, omega)
            + Si(Dωf,omega)  # Dωf = β1*omega
            - Si(dkdomegadx, omega)
            ==
            Source(Pω)
            # + Source(dkdomegadx)
    ) → eqn

    
    # Set up preconditioners

    @reset kl_eqn.preconditioner = set_preconditioner(solvers.kl.preconditioner, kl_eqn)
    @reset k_eqn.preconditioner = set_preconditioner(solvers.k.preconditioner, k_eqn)
    @reset ω_eqn.preconditioner = set_preconditioner(solvers.omega.preconditioner, ω_eqn)
    
    # preallocating solvers

    @reset kl_eqn.solver = _workspace(solvers.kl.solver, _b(kl_eqn))
    @reset k_eqn.solver = _workspace(solvers.k.solver, _b(k_eqn))
    @reset ω_eqn.solver = _workspace(solvers.omega.solver, _b(ω_eqn))

    TF = _get_float(mesh)
    time = zero(TF) # assuming time=0
    grad!(∇ω, omegaf, omega, boundaries.omega, time, config)
    grad!(∇k, kf, k, boundaries.k, time, config)

    # Wall distance calculation
    new_config = wall_distance!(model, wallBCs, config)

    init_residuals = (:k, 1.0),(:kl, 1.0),(:omega, 1.0)
    init_convergence = false
    state = ModelState(init_residuals, init_convergence)

    return KOmegaLKEModel(
        turbulence,
        k_eqn,
        ω_eqn,
        kl_eqn,
        nueffkLS,
        nueffkS,
        nueffωS,
        nuL,
        nuts,
        Ω,
        γ,
        fv,
        normU,
        Reυ,
        ∇k,
        ∇ω,
        state
    ), new_config
end

# Model solver call (implementation)
"""
   turbulence!(rans::KOmegaLKEModel, model::Physics{T,F,SO,M,Turb,E,D,BI}, S, prev, time, config
    ) where {T,F,SO,M,Turb<:AbstractTurbulenceModel,E,D,BI}

Run turbulence model transport equations.

### Input
- `rans::KOmegaLKEModel` -- KOmega turbulence model.
- `model`  -- Physics model defined by user.
- `S`   -- Strain rate tensor.
- `prev`  -- Previous field.
- `time`   -- 
- `config` -- Configuration structure defined by user with solvers, schemes, runtime and 
              hardware structures set.

"""
function turbulence!(
    rans::KOmegaLKEModel, model::Physics{T,F,SO,M,Turb,E,D,BI}, S, prev, time, config
    ) where {T,F,SO,M,Turb<:AbstractTurbulenceModel,E,D,BI}
    mesh = model.domain
    (; momentum) = model
    (; k, omega, kl, nut, y, kf, omegaf, klf, nutf, coeffs, Tu) = rans.turbulence
    (; nu) = model.fluid
    (; U, Uf, gradU) = S
    
    (; k_eqn, ω_eqn, kl_eqn, nueffkLS, nueffkS, nueffωS, nuL, nuts, Ω, γ, fv, ∇k, ∇ω, normU, Reυ, state) = rans
    (; solvers, runtime, boundaries) = config

    nueffkL = get_flux(kl_eqn, 3)
    DkLf = get_flux(kl_eqn, 4)
    PkL = get_source(kl_eqn, 1)

    nueffk = get_flux(k_eqn, 3)
    Dkf = get_flux(k_eqn, 4)
    Pk = get_source(k_eqn, 1)

    nueffω = get_flux(ω_eqn, 3)
    Dωf = get_flux(ω_eqn, 4)
    dkdomegadx = get_flux(ω_eqn, 5) # cross diffusion term
    Pω = get_source(ω_eqn, 1)
    # dkdomegadx = get_source(ω_eqn, 2) # cross diffusion term

    # Get kMin for safety checks (assume it's available in config or model)
    # If not available, you'll need to add it to the model structure
    # For now, using a typical value
    kMin = 1e-10  # Adjust based on your actual kMin_ value

    # 1. Update Velocity Gradients
    grad!(gradU, Uf, U, boundaries.U, time, config) 
    limit_gradient!(config.schemes.U.limiter, gradU, U, config)

    # =========================================================================
    # STEP 2: Setup Laminar Kinetic Energy (kl) Equation 
    # =========================================================================
    xcal_foreach(kl, config) do i
        g = gradU[i]
        
        # Compressible Trace
        divU = g[1,1] + g[2,2] + g[3,3]
        
        # Extract Strain Rate Squared (S2)
        sum_S2 = 0.0
        for j ∈ 1:3, k ∈ 1:3
            symm_jk = 0.5 * (g[j,k] + g[k,j])
            if j == k; symm_jk -= (1.0 / 3.0) * divU; end
            sum_S2 += symm_jk * symm_jk
        end
        S2_val = 2.0 * sum_S2
        
        # Calculate local magnitude of U directly from the vector
        u = U[i]
        normU_i = sqrt(u[1]^2 + u[2]^2 + u[3]^2)
        normU[i] = normU_i
        
        # Extract frequently reused variables to registers
        nu_i = nu[i]
        y_i  = y[i]
        kl_i = kl[i]
        
        # FIXED: Added max(normU, sqrt(kMin)) as in OpenFOAM
        ReLambda = max(normU_i, sqrt(kMin)) * y_i / nu_i
        ReUpsilon = (2.0 * nu_i^2 * kl_i / (y_i^2))^0.25 * y_i / nu_i
        Reυ[i] = ReUpsilon
        
        η = coeffs.C1 * tanh(coeffs.C2 * (Tu^coeffs.C3) + coeffs.C4)
        
        # Sources and fluxes for kl
        PkL[i] = sqrt(S2_val) * η * kl_i * ReUpsilon^(-1.30) * ReLambda^0.5
        DkLf[i] = 2.0 * nu_i / (y_i^2)
        nueffkLS[i] = nu_i + coeffs.σkL * sqrt(kl_i) * y_i
    end

    interpolate!(nueffkL, nueffkLS, config)
    correct_boundaries!(nueffkL, nueffkLS, boundaries.nut, time, config)

    # 3. Solve kl equation
    prev .= kl.values
    discretise!(kl_eqn, prev, config)
    apply_boundary_conditions!(kl_eqn, boundaries.kl, nothing, time, config)
    implicit_relaxation!(kl_eqn, kl.values, solvers.kl.relax, nothing, config)
    update_preconditioner!(kl_eqn.preconditioner, mesh, config)
    kl_res = solve_system!(kl_eqn, solvers.kl, kl, nothing, config)
    bound!(kl, config)

    # =========================================================================
    # STEP 4: Gradients for ω Cross-Diffusion 
    # =========================================================================
    grad!(∇ω, omegaf, omega, boundaries.omega, time, config)
    grad!(∇k, kf, k, boundaries.k, time, config)
    inner_product!(dkdomegadx, ∇k, ∇ω, config)

    # =========================================================================
    # STEP 5: Setup ω and k Equations (Intermittency, fv, sources) 
    # =========================================================================
    xcal_foreach(k, config) do i
        g = gradU[i]
        
        divU = g[1,1] + g[2,2] + g[3,3]
        sum_S2 = 0.0
        sum_Omega2 = 0.0
        for j ∈ 1:3, k ∈ 1:3
            symm_jk = 0.5 * (g[j,k] + g[k,j])
            if j == k; symm_jk -= (1.0 / 3.0) * divU; end
            sum_S2 += symm_jk * symm_jk
            
            skew_jk = 0.5 * (g[j,k] - g[k,j])
            sum_Omega2 += skew_jk * skew_jk
        end
        S2_val = 2.0 * sum_S2
        Omega_val = sqrt(2.0 * sum_Omega2)
        Ω[i] = Omega_val # Store for fSS trigger later
        
        # Extract frequently reused variables
        nu_i = nu[i]
        y_i  = y[i]
        kl_i = kl[i] 
        k_i  = k[i]
        omega_i = omega[i]
        normU_i = normU[i]
        
        safe_omega = max(omega_i, 1e-15)
        safe_nu    = max(nu_i, 1e-15)

        # Recalculate intermediate nutL1 for γ trigger
        # FIXED: Added max(normU, sqrt(kMin))
        ReLambda = max(normU_i, sqrt(kMin)) * y_i / nu_i
        ReUpsilon = (2.0 * nu_i^2 * kl_i / (y_i^2))^0.25 * y_i / nu_i
        η = coeffs.C1 * tanh(coeffs.C2 * (Tu^coeffs.C3) + coeffs.C4)
        nutL1 = η * kl_i * sqrt(S2_val) * ReUpsilon^(-1.30) * ReLambda^0.5 / max(S2_val, (normU_i/y_i)^2)
        
        # Intermittency Trigger (γ) matching OpenFOAM ReL constraints
        ReL = min(kl_i / max(min(nu_i, nutL1), 1e-15) / max(Omega_val, 1e-10), 5000.0)
        gamma_val = min(ReL^2, coeffs.Ccrit) / coeffs.Ccrit
        γ[i] = gamma_val
        
        # Damping Function (fv)
        fv_val = 1.0 - exp(-sqrt(k_i / (safe_nu * safe_omega)) / coeffs.Cv)
        fv[i] = fv_val
        
        # Update ω Terms
        # Production for omega: Cω1 * Pk where Pk = S2 (no eddy viscosity needed - k/omega and omega/k cancel)
        Pω[i] = coeffs.Cω1 * S2_val
        Dωf[i] = coeffs.Cω2 * omega_i
        nueffωS[i] = nu_i + coeffs.σω * (k_i / safe_omega) * 1.0 
        
        # Safely compute cross-diffusion divided by ω^2
        dkdomega_val = dkdomegadx[i]
        dkdomegadx[i] = max((coeffs.σd / (safe_omega^2)) * dkdomega_val, 0.0)
        
        # Update k Terms
        # FIXED: Added (k_i / safe_omega) factor for turbulent eddy viscosity
        # Production = fv * (k/omega) * S2 * gamma
        Pk[i] = fv_val * (k_i / safe_omega) * S2_val * gamma_val
        Dkf[i] = coeffs.Cμ * omega_i * gamma_val
        nueffkS[i] = nu_i + coeffs.σk * (k_i / safe_omega) * 1.0 
    end

    interpolate!(nueffω, nueffωS, config)
    correct_boundaries!(nueffω, nueffωS, boundaries.nut, time, config)
    
    interpolate!(nueffk, nueffkS, config)
    correct_boundaries!(nueffk, nueffkS, boundaries.nut, time, config)
    correct_production!(Pk, boundaries.k, model, S.gradU, config)

    # 6. Solve omega equation
    prev .= omega.values
    discretise!(ω_eqn, prev, config)
    apply_boundary_conditions!(ω_eqn, boundaries.omega, nothing, time, config)
    implicit_relaxation!(ω_eqn, omega.values, solvers.omega.relax, nothing, config)
    constrain_equation!(ω_eqn, boundaries.omega, model, config) 
    update_preconditioner!(ω_eqn.preconditioner, mesh, config)
    ω_res = solve_system!(ω_eqn, solvers.omega, omega, nothing, config)
    bound!(omega, config)

    # 7. Solve k equation
    prev .= k.values
    discretise!(k_eqn, prev, config)
    apply_boundary_conditions!(k_eqn, boundaries.k, nothing, time, config)
    implicit_relaxation!(k_eqn, k.values, solvers.k.relax, nothing, config)
    update_preconditioner!(k_eqn.preconditioner, mesh, config)
    k_res = solve_system!(k_eqn, solvers.k, k, nothing, config)
    bound!(k, config)

    # =========================================================================
    # STEP 8: Post-Solve Update (nuL, fSS, nut)
    # =========================================================================
    xcal_foreach(nut, config) do i
        g = gradU[i]
        
        divU = g[1,1] + g[2,2] + g[3,3]
        sum_S2 = 0.0
        sum_Omega2 = 0.0
        for j ∈ 1:3, k ∈ 1:3
            symm_jk = 0.5 * (g[j,k] + g[k,j])
            if j == k; symm_jk -= (1.0 / 3.0) * divU; end
            sum_S2 += symm_jk * symm_jk
            
            skew_jk = 0.5 * (g[j,k] - g[k,j])
            sum_Omega2 += skew_jk * skew_jk
        end
        S2_val = 2.0 * sum_S2
        Omega_val = sqrt(2.0 * sum_Omega2)
        
        # Extract frequently reused variables
        nu_i = nu[i]
        y_i  = y[i]
        kl_i = kl[i]
        k_i  = k[i]
        omega_i = omega[i]
        normU_i = normU[i]
        
        # FIXED: Added max(normU, sqrt(kMin))
        ReLambda = max(normU_i, sqrt(kMin)) * y_i / nu_i
        ReUpsilon = (2.0 * nu_i^2 * kl_i / (y_i^2))^0.25 * y_i / nu_i
        η = coeffs.C1 * tanh(coeffs.C2 * (Tu^coeffs.C3) + coeffs.C4)
        
        # Update Laminar contribution nuL
        PkL_val = sqrt(S2_val) * η * kl_i * ReUpsilon^(-1.30) * ReLambda^0.5
        nuL_val = PkL_val / max(S2_val, (normU_i/y_i)^2)
        nuL[i] = nuL_val
        
        # Update fSS (Uses Omega to evaluate stagnation points properly)
        fSS = exp(-(coeffs.CSS * nu_i * Omega_val / max(k_i, 1e-15))^2)
        nuts_val = fSS * (k_i / max(omega_i, 1e-15))
        nuts[i] = nuts_val
        
        # Final Total Eddy Viscosity
        nut[i] = nuts_val + nuL_val
    end

    interpolate!(nutf, nut, config)
    correct_boundaries!(nutf, nut, boundaries.nut, time, config)
    correct_eddy_viscosity!(nutf, boundaries.nut, model, config)

    # 9. Update residuals and convergence status
    residuals = ((:k, k_res),(:kl, kl_res),(:omega, ω_res))
    k_converged  = k_res  < solvers.k.convergence
    kl_converged = kl_res < solvers.kl.convergence
    ω_converged  = ω_res  < solvers.omega.convergence
    converged = k_converged && kl_converged && ω_converged
    state.residuals = residuals
    state.converged = converged
    
    return nothing
end
# function turbulence!(
#     rans::KOmegaLKEModel, model::Physics{T,F,SO,M,Turb,E,D,BI}, S, prev, time, config
#     ) where {T,F,SO,M,Turb<:AbstractTurbulenceModel,E,D,BI}
#     mesh = model.domain
#     (; momentum) = model
#     U = momentum.U
#     (; k, omega, kl, nut, y, kf, omegaf, klf, nutf, coeffs, Tu) = rans.turbulence
#     (; nu) = model.fluid
#     (; U, Uf, gradU) = S
    
#     (; k_eqn, ω_eqn, kl_eqn, nueffkLS, nueffkS, nueffωS, nuL, nuts, Ω, γ, fv, ∇k, ∇ω, normU, Reυ, state) = rans
#     (; solvers, runtime, boundaries) = config

#     nueffkL = get_flux(kl_eqn, 3)
#     DkLf = get_flux(kl_eqn, 4)
#     PkL = get_source(kl_eqn, 1)

#     nueffk = get_flux(k_eqn, 3)
#     Dkf = get_flux(k_eqn, 4)
#     Pk = get_source(k_eqn, 1)

#     nueffω = get_flux(ω_eqn, 3)
#     Dωf = get_flux(ω_eqn, 4)
#     dkdomegadx = get_flux(ω_eqn, 5)
#     Pω = get_source(ω_eqn, 1)
#     # dkdomegadx = get_source(ω_eqn, 2) # cross diffusion term


#     grad!(gradU, Uf, U, boundaries.U, time, config) # must update before calculating S
#     limit_gradient!(config.schemes.U.limiter, gradU, U, config)
#     magnitude2!(Pk, S, config, scale_factor=2.0)
#     magnitude2!(Ω, S, config, scale_factor=2.0) # 
#     S2 = Ω # using Ω to store S^2 (temporary - needs cleaning up!)

#     # Update kl fluxes and terms
#     magnitude!(normU, U, config)

#     ReLambda = @. max(normU.values, sqrt(eps()))*y.values/nu.values
#     @. Reυ.values = (2*nu.values^2*kl.values/(y.values^2))^0.25*y.values/nu.values;
#     η = coeffs.C1*tanh(coeffs.C2*(Tu^coeffs.C3)+coeffs.C4)
#     @. PkL.values = sqrt(S2.values)*η*kl.values*Reυ.values^(-1.30)*ReLambda^(0.5)
#     @. DkLf.values = (2*nu.values)/(y.values^2)
#     @. nueffkLS.values = nu.values+(coeffs.σkL*sqrt(kl.values)*y.values)
#     interpolate!(nueffkL, nueffkLS, config)
#     correct_boundaries!(nueffkL, nueffkLS, boundaries.nut, time, config)

#     # Solve kl equation
#     prev .= kl.values
#     discretise!(kl_eqn, prev, config)
#     apply_boundary_conditions!(kl_eqn, boundaries.kl, nothing, time, config)
#     implicit_relaxation!(kl_eqn, kl.values, solvers.kl.relax, nothing, config)
#     update_preconditioner!(kl_eqn.preconditioner, mesh, config)
#     kl_res = solve_system!(kl_eqn, solvers.kl, kl, nothing, config)
#     bound!(kl, config)

#     #Damping and trigger
#     # @. fv.values = 1-exp(-sqrt(k.values/(nu.values*omega.values))/coeffs.Cv)
#     ReL = 
#     @. γ.values = min(
#             (kl.values/(min(nu.values,nuL.values)*sqrt(S2.values)))^2,
#             coeffs.Ccrit
#         )/coeffs.Ccrit
#     fSS = @. exp(-(coeffs.CSS*nu.values*sqrt(S2.values)/k.values)^2) # should be Ω but S works

#     #Update ω fluxes
#     # double_inner_product!(Pk, S, gradU) # multiplied by 2 (def of Sij) (Pk = S² at this point)
#     # interpolate!(kf, k, config)
#     # correct_boundaries!(nutf, k, boundaries.k, time, config)
#     # correct_boundaries!(kf, k, boundaries.k, time, config)
#     # interpolate!(omegaf, omega, config)
#     # correct_boundaries!(nutf, omega, boundaries.omega, time, config)
#     # correct_boundaries!(nutf, omega, boundaries.omega, time, config)
#     grad!(∇ω, omegaf, omega, boundaries.omega, time, config)
#     grad!(∇k, kf, k, boundaries.k, time, config)
#     inner_product!(dkdomegadx, ∇k, ∇ω, config)
#     # @. Pω.values = coeffs.Cω1 * Pk.values * nut.values * (omega.values / k.values)
#     @. Pω.values = coeffs.Cω1 * Pk.values # * nut.values * (omega.values / k.values)

#     # @. dkdomegadx.values = max((coeffs.σd / omega.values) * dkdomegadx.values, 0.0)
#     @. dkdomegadx.values = max((coeffs.σd / omega.values^2) * dkdomegadx.values, 0.0)

#     # Use below if calculating cross-diffusion as implicit source
#     # @. dkdomegadx.values = max((coeffs.σd / omega.values^2) * dkdomegadx.values, 0.0) 
#     @. Dωf.values = coeffs.Cω2 * omega.values
#     # @. nueffωS.values = nu.values+(coeffs.σω*nut_turb.values*γ.values)
#     @. nueffωS.values = nu.values+(coeffs.σω*(k.values/omega.values)*γ.values)
#     interpolate!(nueffω, nueffωS, config)
#     correct_boundaries!(nueffω, nueffωS, boundaries.nut, time, config)

#     #Update k fluxes
#     @. Dkf.values = coeffs.Cμ*omega.values*γ.values
#     @. nueffkS.values = nu.values+(coeffs.σk*(k.values/omega.values)*γ.values)
#     interpolate!(nueffk, nueffkS, config)
#     correct_boundaries!(nueffk, nueffkS, boundaries.nut, time, config)
#     # @. Pk.values = nut.values*Pk.values*γ.values*fv.values

#     @. fv.values = 1-exp(-sqrt(k.values/(nu.values*omega.values))/coeffs.Cv) # moved here
#     @. Pk.values = k.values/omega.values*Pk.values*γ.values*fv.values
#     correct_production!(Pk, boundaries.k, model, S.gradU, config)

#     # Solve omega equation
#     prev .= omega.values
#     discretise!(ω_eqn, prev, config)
#     apply_boundary_conditions!(ω_eqn, boundaries.omega, nothing, time, config)
#     implicit_relaxation!(ω_eqn, omega.values, solvers.omega.relax, nothing, config)
#     constrain_equation!(ω_eqn, boundaries.omega, model, config) # active with WFs only
#     update_preconditioner!(ω_eqn.preconditioner, mesh, config)
#     ω_res = solve_system!(ω_eqn, solvers.omega, omega, nothing, config)
#     # constrain_boundary!(omega, boundaries.omega, model, config) # active with WFs only
#     bound!(omega, config)

#     # Solve k equation
#     prev .= k.values
#     discretise!(k_eqn, prev, config)
#     apply_boundary_conditions!(k_eqn, boundaries.k, nothing, time, config)
#     implicit_relaxation!(k_eqn, k.values, solvers.k.relax, nothing, config)
#     update_preconditioner!(k_eqn.preconditioner, mesh, config)
#     k_res = solve_system!(k_eqn, solvers.k, k, nothing, config)
#     bound!(k, config)

#     # grad!(∇ω, omegaf, omega, boundaries.omega, time, config)
#     # grad!(∇k, kf, k, boundaries.k, time, config)

#     # @. nut_turb.values = k.values/omega.values
#     # ReLambda = @. normU.values*y.values/nu.values
#     ReLambda = @. max(normU.values, sqrt(eps()))*y.values/nu.values

#     @. Reυ.values = (2*nu.values^2*kl.values/(y.values^2))^0.25*y.values/nu.values;
#     @. PkL.values = sqrt(S2.values)*η*kl.values*Reυ.values^(-1.30)*ReLambda^(0.5) # update
#     @. nuL.values = PkL.values/max(S2.values,(normU.values/y.values)^2)

#     fSS = @. exp(-(coeffs.CSS*nu.values*sqrt(S2.values)/k.values)^2) # should be Ω but S works
#     @. nuts.values = fSS*(k.values/omega.values)
#     @. nut.values = nuts.values + nuL.values

#     interpolate!(nutf, nut, config)
#     correct_boundaries!(nutf, nut, boundaries.nut, time, config)
#     correct_eddy_viscosity!(nutf, boundaries.nut, model, config)

#     # update residuals and convergence status
#     residuals = ((:k, k_res),(:kl, kl_res),(:omega, ω_res))
#     k_converged = k_res < solvers.k.convergence
#     kl_converged = kl_res < solvers.kl.convergence
#     ω_converged = ω_res < solvers.omega.convergence
#     converged = k_converged && kl_converged && ω_converged
#     state.residuals = residuals
#     state.converged = converged
#     return nothing
# end

# Specialise VTK writer
function save_output(model::Physics{T,F,SO,M,Tu,E,D,BI}, outputWriter, iteration, time, config
    ) where {T,F,SO,M,Tu<:KOmegaLKE,E,D,BI}
    args = (
        ("U", model.momentum.U), 
        ("p", model.momentum.p),
        ("k", model.turbulence.k),
        ("omega", model.turbulence.omega),
        ("kl", model.turbulence.kl),
        ("nut", model.turbulence.nut),
        ("y", model.turbulence.y)
    )
    write_results(iteration, time, model.domain, outputWriter, config.boundaries, args...)
end