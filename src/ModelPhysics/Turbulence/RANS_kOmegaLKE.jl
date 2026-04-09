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
struct KOmegaLKE{S1,S2,S3,S4,F1,F2,F3,F4,C1,C2,Y} <: AbstractRANSModel 
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
end 
Adapt.@adapt_structure KOmegaLKE

# Model type definition (hold equation definitions and internal data)
struct KOmegaLKEModel{
    T,E1,E2,E3,F1,F2,F3,S1,S2,S3,S4,S5,S6,S7,S8,S9,S10,V1,V2,State}
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
    divU::S8
    S2::S9
    ReLambda::S10
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
    Tu = rans.args.Tu
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
        σω = 0.5,
        η = 0.02974 * tanh(59.79 * (Tu^1.191) + 1.65e-13)
    )

    # Allocate wall distance "y" and setup boundary conditions
    y = ScalarField(mesh)
    KOmegaLKE(k, omega, kl, nut, kf, omegaf, klf, nutf, coeffs, Tu, y)
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
        divU,
        S2,
        ReLambda,
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
    (; k, omega, kl, kf, omegaf, klf, y) = model.turbulence
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
    divU = ScalarField(mesh)
    S2_field = ScalarField(mesh)
    ReLambda = ScalarField(mesh)
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

    @reset kl_eqn.solver = _workspace(solvers.kl.solver, _A(kl_eqn), _b(kl_eqn))
    @reset k_eqn.solver = _workspace(solvers.k.solver, _A(k_eqn), _b(k_eqn))
    @reset ω_eqn.solver = _workspace(solvers.omega.solver, _A(ω_eqn), _b(ω_eqn))

    TF = _get_float(mesh)
    time = zero(TF) # assuming time=0
    grad!(∇ω, omegaf, omega, boundaries.omega, time, config)
    grad!(∇k, kf, k, boundaries.k, time, config)

    # Wall distance calculation
    new_config = wall_distance!(model, model.wall_info, config)


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
        divU,
        S2_field,
        ReLambda,
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
    
    (; k_eqn, ω_eqn, kl_eqn, nueffkLS, nueffkS, nueffωS, nuL, nuts, Ω, γ, ∇k, ∇ω, normU, divU, S2, ReLambda, state) = rans
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

    kMin = eps()  # set kMin_ value

    # Update Velocity Gradient and Calculate Flow Quantities (ONCE)
    grad!(gradU, Uf, U, boundaries.U, time, config) 
    limit_gradient!(config.schemes.U.limiter, gradU, U, config)

    η = coeffs.η

    xcal_foreach(k, config) do i
        g = gradU[i]

        # Velocity divergence
        divU_val = tr(g)
        divU[i] = divU_val

        # Calculate strain rate, vorticity & production
        S_dev = 0.5*(g + g') - divU_val/3*I # Dev(S)
        S2[i] = 2.0 * sum(S_dev.^2) # S2 = 2*magSqr(dev(symm(gradU)))
        Ω[i] = sqrt(2.0 * sum((0.5*(g - g')).^2)) # Omega = sqrt(2)*mag(skew(gradU))
        Pk[i] = sum(g .* 2*S_dev) # Pk = gradU && dev(twoSymm(gradU))

        # Calculate velocity magnitude
        u = U[i]
        normU[i] = sqrt(u[1]^2 + u[2]^2 + u[3]^2)

        # Cache ReLambda (used in gamma, kl, and nut loops)
        ReLambda[i] = max(normU[i], sqrt(kMin)) * y[i] / nu[i]
    end

    # Calculate gamma, PkL, DkLf, nueffkLS (fused gamma + kl setup)
    xcal_foreach(γ, config) do i
        nu_i = nu[i]
        y_i = y[i]
        kl_i = kl[i]
        S2_val = S2[i]
        Omega_val = Ω[i]
        normU_i = normU[i]
        ReLambda_val = ReLambda[i]

        # Reynolds numbers
        ReUpsilon = sqrt(sqrt(2.0 * nu_i^2 * kl_i / (y_i^2))) * y_i / nu_i

        # nutL1 for ReL calculation
        nutL1 = η * kl_i * sqrt(S2_val) * ReUpsilon^(-1.30) * ReLambda_val^0.5 / max(S2_val, (normU_i/y_i)^2)

        # Intermittency trigger gamma
        ReL = min(kl_i / max(min(nu_i, nutL1), 1e-15) / max(Omega_val, 1e-15), 5000.0)
        γ[i] = min(ReL^2, coeffs.Ccrit) / coeffs.Ccrit

        # kl equation setup
        PkL[i] = sqrt(S2_val) * η * kl_i * ReUpsilon^(-1.30) * ReLambda_val^0.5
        DkLf[i] = 2.0 * nu_i / (y_i^2)
        nueffkLS[i] = nu_i + coeffs.σkL * sqrt(kl_i) * y_i
    end

    interpolate!(nueffkL, nueffkLS, config)
    correct_boundaries!(nueffkL, nueffkLS, boundaries.nut, time, config)

    # Solve kl equation
    prev .= kl.values
    discretise!(kl_eqn, prev, config)
    apply_boundary_conditions!(kl_eqn, boundaries.kl, nothing, time, config)
    implicit_relaxation!(kl_eqn, kl.values, solvers.kl.relax, nothing, config)
    update_preconditioner!(kl_eqn.preconditioner, mesh, config)
    kl_res = solve_system!(kl_eqn, solvers.kl, kl, nothing, config)
    bound!(kl, config)

    # Calculate Gradients for Cross-Diffusion  
    grad!(∇ω, omegaf, omega, boundaries.omega, time, config)
    grad!(∇k, kf, k, boundaries.k, time, config)
    inner_product!(dkdomegadx, ∇k, ∇ω, config)

    # Setup and Solve omega Equation
    xcal_foreach(omega, config) do i
        omega_i = omega[i] 
        safe_omega = max(omega_i, 1e-15)
        
        Pω[i] = coeffs.Cω1 * Pk[i] # production
        Pω[i] -= (2.0/3.0) * coeffs.Cω1 * divU[i] * omega_i # desctruction
        Dωf[i] = coeffs.Cω2 * omega_i # dissipation
        nueffωS[i] = nu[i] + coeffs.σω * (k[i] / safe_omega) # diffusion
        dkdomegadx[i] = max((coeffs.σd / (safe_omega^2)) * dkdomegadx[i], 0.0) # x-diffusion
    end

    interpolate!(nueffω, nueffωS, config)
    correct_boundaries!(nueffω, nueffωS, boundaries.nut, time, config)

    # Solve omega equation
    prev .= omega.values
    discretise!(ω_eqn, prev, config)
    apply_boundary_conditions!(ω_eqn, boundaries.omega, nothing, time, config)
    implicit_relaxation!(ω_eqn, omega.values, solvers.omega.relax, nothing, config)
    constrain_equation!(ω_eqn, boundaries.omega, model, config) 
    update_preconditioner!(ω_eqn.preconditioner, mesh, config)
    ω_res = solve_system!(ω_eqn, solvers.omega, omega, nothing, config)
    bound!(omega, config)

    # Calculate fv and setup k equation (fused)
    xcal_foreach(k, config) do i
        omega_i = omega[i]
        gamma_val = γ[i]
        safe_omega = max(omega_i, 1e-15)
        safe_k = max(k[i], 1e-15)
        safe_nu = max(nu[i], 1e-15)

        # fv calculation
        fv_val = 1.0 - exp(-sqrt(safe_k / (safe_nu * safe_omega)) / coeffs.Cv)

        # Production with limiter
        Pk_unlimited = fv_val * (safe_k / safe_omega) * Pk[i] * gamma_val
        Pk_limited = min(Pk_unlimited, 20.0 * coeffs.Cμ * safe_k * omega_i)
        Pk[i] = Pk_limited - (2.0/3.0) * divU[i] * k[i]

        # Destruction
        Dkf[i] = coeffs.Cμ * gamma_val * omega_i

        # Diffusion
        nueffkS[i] = nu[i] + coeffs.σk * (safe_k / safe_omega)
    end

    interpolate!(nueffk, nueffkS, config)
    correct_boundaries!(nueffk, nueffkS, boundaries.nut, time, config)
    correct_production!(Pk, boundaries.k, model, S.gradU, config)

    # Solve k equation
    prev .= k.values
    discretise!(k_eqn, prev, config)
    apply_boundary_conditions!(k_eqn, boundaries.k, nothing, time, config)
    implicit_relaxation!(k_eqn, k.values, solvers.k.relax, nothing, config)
    update_preconditioner!(k_eqn.preconditioner, mesh, config)
    k_res = solve_system!(k_eqn, solvers.k, k, nothing, config)
    bound!(k, config)

    # Calculate Final nutL and nut
    xcal_foreach(nut, config) do i
        kl_i = kl[i]
        nu_i = nu[i]
        y_i = y[i]
        S2_val = S2[i]
        normU_i = normU[i]
        safe_omega = max(omega[i], 1e-15)
        safe_k = max(k[i], 1e-15)
        
        # Calculate nutL
        ReLambda_val = ReLambda[i]
        ReUpsilon = sqrt(sqrt(2.0 * nu_i^2 * kl_i / (y_i^2))) * y_i / nu_i
        PkL_val = sqrt(S2_val) * η * kl_i * ReUpsilon^(-1.30) * ReLambda_val^0.5
        nuL_val = PkL_val / max(S2_val, (normU_i/y_i)^2)
        nuL[i] = nuL_val
        
        # Calculate fSS
        fSS = exp(-(coeffs.CSS * nu_i * Ω[i] / safe_k)^2)
        
        # Turbulent eddy viscosity
        nuts_val = fSS * (safe_k / safe_omega)
        nuts[i] = nuts_val
        
        # Total eddy viscosity
        nut[i] = nuts_val + nuL_val
    end

    interpolate!(nutf, nut, config)
    correct_boundaries!(nutf, nut, boundaries.nut, time, config)
    correct_eddy_viscosity!(nutf, boundaries.nut, model, config)

    # Update Residuals and Convergence Status 
    residuals = ((:k, k_res),(:kl, kl_res),(:omega, ω_res))
    k_converged  = k_res  < solvers.k.convergence
    kl_converged = kl_res < solvers.kl.convergence
    ω_converged  = ω_res  < solvers.omega.convergence
    converged = k_converged && kl_converged && ω_converged
    state.residuals = residuals
    state.converged = converged
    
    return nothing
end

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