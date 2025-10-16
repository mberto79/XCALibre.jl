export KOmegaSST

# Reference:
# Wilcox, D. C., Turbulence Modeling for CFD, 2nd edition, DCW Industries, Inc., La Canada CA, 1998

# Model type definition
"""
    KOmega <: AbstractTurbulenceModel

kOmega model containing all kOmega field parameters.

### Fields
- 'k' -- Turbulent kinetic energy ScalarField.
- 'omega' -- Specific dissipation rate ScalarField.
- 'nut' -- Eddy viscosity ScalarField.
- 'kf' -- Turbulent kinetic energy FaceScalarField.
- 'omegaf' -- Specific dissipation rate FaceScalarField.
- 'nutf' -- Eddy viscosity FaceScalarField.
- 'coeffs' -- Model coefficients.

"""
struct KOmegaSST{S1,S2,S3,F1,F2,F3,C1,C2,C3,Y,BC} <: AbstractRANSModel
    k::S1
    omega::S2
    nut::S3
    kf::F1
    omegaf::F2
    nutf::F3
    coeffs::C1
    gamma1::C2
    gamma2::C3
    y::Y
    wallBCs::BC
end
Adapt.@adapt_structure KOmegaSST

struct KOmegaSSTModel{E1,E2,S,S1,F1,F2,S2,S3,S4,S5,S6,F3,S7,S8,V1,V2}
    k_eqn::E1 
    ω_eqn::E2
    state::S
    β::S1
    σkf::F1
    σωf::F2
    γ::S2
    CDkω::S3
    arg1::S4
    F1::S5
    F1f::F3
    arg2::S6
    F2::S7
    Ω::S8
    ∇k::V1
    ∇ω::V2
end
Adapt.@adapt_structure KOmegaSSTModel

# Model API constructor (pass user input as keyword arguments and process as needed)
RANS{KOmegaSST}(; β⁺=0.09, α1=0.31, σk1=0.85, σk2=1.0, σω1=0.5, σω2=0.856, β1=0.075, β2=0.0828, κ=0.41, walls) = begin 
    coeffs = (β⁺=β⁺, α1=α1, σk1=σk1, σk2=σk2, σω1=σω1, σω2=σω2, β1=β1, β2=β2, κ=κ, walls=walls)
    ARG = typeof(coeffs)
    RANS{KOmegaSST,ARG}(coeffs)
end

# Functor as constructor (internally called by Physics API): Returns fields and user data
(rans::RANS{KOmegaSST, ARG})(mesh) where ARG = begin
    k = ScalarField(mesh)
    omega = ScalarField(mesh)
    nut = ScalarField(mesh)
    kf = FaceScalarField(mesh)
    omegaf = FaceScalarField(mesh)
    nutf = FaceScalarField(mesh)
    coeffs = rans.args
    gamma1 = (coeffs.β1/coeffs.β⁺) - coeffs.σω1*coeffs.κ^2/sqrt(coeffs.β⁺)
    gamma2 = (coeffs.β2/coeffs.β⁺) - coeffs.σω2*coeffs.κ^2/sqrt(coeffs.β⁺)

    # Allocate wall distance "y" and setup boundary conditions
    y = ScalarField(mesh)
    wallBCs = rans.args.walls
   

    KOmegaSST(k, omega, nut, kf, omegaf, nutf, coeffs, gamma1, gamma2, y, wallBCs) 
end

# Model initialisation
"""
    initialise(turbulence::KOmega, model::Physics{T,F,SO,M,Tu,E,D,BI}, mdotf, peqn, config
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
- `KOmegaModel(k_eqn, ω_eqn)`  -- Turbulence model structure.

"""
function initialise(
    turbulence::KOmegaSST, model::Physics{T,F,SO,M,Tu,E,D,BI}, mdotf, peqn, config
    ) where {T,F,SO,M,Tu,E,D,BI}

    (; solvers, schemes, runtime, hardware) = config
    (; k, omega, nut, wallBCs) = turbulence
    

    (; rho) = model.fluid
    mesh = mdotf.mesh
    eqn = peqn.equation

    # define fluxes and sources
    mueffk = FaceScalarField(mesh)
    mueffω = FaceScalarField(mesh)
    Dkf = ScalarField(mesh)
    Dωf = ScalarField(mesh)
    Pk = ScalarField(mesh)
    Pω = ScalarField(mesh)
    dkdomegadx = ScalarField(mesh)

    CDkω = ScalarField(mesh)
    arg1 = ScalarField(mesh)
    F1 = ScalarField(mesh)
    F1f = FaceScalarField(mesh)

    arg2 = ScalarField(mesh)
    F2 = ScalarField(mesh)

    β = ScalarField(mesh)
    σkf = FaceScalarField(mesh)
    σωf = FaceScalarField(mesh)
    γ = ScalarField(mesh)

    Ω = ScalarField(mesh)

    ∇k = Grad{schemes.k.gradient}(k)
    ∇ω = Grad{schemes.p.gradient}(omega)
    
    k_eqn = (
            Time{schemes.k.time}(rho, k)
            + Divergence{schemes.k.divergence}(mdotf, k) 
            - Laplacian{schemes.k.laplacian}(mueffk, k) 
            + Si(Dkf,k) # Dkf = β⁺rho*omega
            ==
            Source(Pk)
        ) → eqn
    
    ω_eqn = (
            Time{schemes.omega.time}(rho, omega)
            + Divergence{schemes.omega.divergence}(mdotf, omega) 
            - Laplacian{schemes.omega.laplacian}(mueffω, omega) 
            + Si(Dωf,omega)  # Dωf = rho*β1*omega
            + Si(dkdomegadx, omega)
            ==
            Source(Pω) #- Source(dkdomegadx)
    ) → eqn

    # Set up preconditioners
    @reset k_eqn.preconditioner = set_preconditioner(solvers.k.preconditioner, k_eqn)
    @reset ω_eqn.preconditioner = set_preconditioner(solvers.omega.preconditioner, ω_eqn)
    
    # preallocating solvers
    @reset k_eqn.solver = _workspace(solvers.k.solver, _b(k_eqn))
    @reset ω_eqn.solver = _workspace(solvers.omega.solver, _b(ω_eqn))

    new_config = wall_distance!(model, wallBCs, config)

    initial_residual = ((:k, 1.0),(:omega, 1.0))
    return KOmegaSSTModel(k_eqn, ω_eqn, ModelState(initial_residual, false), β, σkf, σωf, γ, CDkω, arg1, F1, F1f, arg2, F2, Ω, ∇k, ∇ω), new_config
end

# Model solver call (implementation)
"""
    turbulence!(rans::KOmegaModel{E1,E2,S1}, model::Physics{T,F,SO,M,Tu,E,D,BI}, S, prev, time, config
    ) where {T,F,SO,M,Tu<:KOmega,E,D,BI,E1,E2,S1}

Run turbulence model transport equations.

### Input
- `rans::KOmegaModel{E1,E2,S1}` -- KOmega turbulence model.
- `model`  -- Physics model defined by user.
- `S`   -- Strain rate tensor.
- `prev`  -- Previous field.
- `time`   -- 
- `config` -- Configuration structure defined by user with solvers, schemes, runtime and 
              hardware structures set.

"""
function turbulence!(
    rans::KOmegaSSTModel{E1,E2,S1}, model::Physics{T,F,SO,M,Tu,E,D,BI}, S, prev, time, config
    ) where {T,F,SO,M,Tu<:AbstractTurbulenceModel,E,D,BI,E1,E2,S1}

    mesh = model.domain
    
    (; rho, rhof, nu, nuf) = model.fluid
    (;k, omega, nut, kf, omegaf, nutf, coeffs, gamma1, gamma2, y) = model.turbulence
    (; U, Uf, gradU) = S
    (;k_eqn, ω_eqn, state, β, σkf, σωf, γ, CDkω, arg1, F1, F1f, arg2, F2, Ω, ∇k, ∇ω) = rans
    (; solvers, runtime, boundaries) = config

    mueffk = get_flux(k_eqn, 3)
    Dkf = get_flux(k_eqn, 4)
    Pk = get_source(k_eqn, 1)

    mueffω = get_flux(ω_eqn, 3)
    Dωf = get_flux(ω_eqn, 4)
    Pω = get_source(ω_eqn, 1)
    # dkdomegadx = get_source(ω_eqn, 2)
    dkdomegadx = get_flux(ω_eqn, 5)

    # update fluxes and sources

    grad!(gradU, Uf, U, boundaries.U, time, config)
    limit_gradient!(config.schemes.U.limiter, gradU, U, config)
    magnitude2!(Pk, S, config, scale_factor=2.0) # multiplied by 2 (def of Sij)
    magnitude2!(Ω, Vorticity(U, gradU), config, scale_factor=2.0) # 
    # @. Ω.values = sqrt(Ω.values) # This is for the proper NASA formulation
    @. Ω.values = sqrt(Pk.values) # gives better comparison with OF

    # interpolate!(kf, k, config)
    # correct_boundaries!(kf, k, boundaries.k, time, config) # Bug here but no issue
    # interpolate!(omegaf, omega, config)
    # correct_boundaries!(omegaf, omega, boundaries.omega, time, config)  # same here
    grad!(∇ω, omegaf, omega, boundaries.omega, time, config)
    grad!(∇k, kf, k, boundaries.k, time, config)
    inner_product!(dkdomegadx, ∇k, ∇ω, config)

    @. CDkω.values = max(2*coeffs.σω2*dkdomegadx.values/omega.values, 1e-10)

    @. arg1.values = min( min(
            max(
                sqrt(max(k.values, eps()))/(coeffs.β⁺*omega.values*y.values), 
                500*nu.values/(omega.values*y.values^2)
                ),
            4*coeffs.σω2*k.values/(CDkω.values*y.values^2)),
         10.0
             )
    

    @. arg2.values = min(max(
            2*sqrt(max(k.values, eps()))/(coeffs.β⁺*omega.values*y.values), 
            500*nu.values/(y.values^2*omega.values)) ,
        100.0
        )

    @. F2.values = tanh(arg2.values^2)
    @. F1.values = tanh(arg1.values^4)
    interpolate!(F1f, F1, config)


    @. σkf.values = coeffs.σk1*F1f.values + (1.0 - F1f.values)*coeffs.σk2
    @. σωf.values = coeffs.σω1*F1f.values + (1.0 - F1f.values)*coeffs.σω2
    @. β.values = coeffs.β1*F1.values + (1.0 - F1.values)*coeffs.β2
    # Here I'm using hard-coded values - need to revert to proper defs used above
    @. γ.values = 5/9*F1.values + (1.0 - F1.values)*0.44 # Chris: revert if you want

    @. mueffω.values = rhof.values * (nuf.values + σωf.values*nutf.values)
    @. mueffk.values = rhof.values * (nuf.values + σkf.values*nutf.values)

    @. Dωf.values = rho.values*β.values*omega.values
    @. Dkf.values = rho.values*coeffs.β⁺*omega.values

    # Production limiter
    @. Pω.values = rho.values*γ.values*min(
        Pk.values,
        (10/coeffs.α1)*coeffs.β⁺*omega.values*max(
            coeffs.α1*omega.values, F2.values*Ω.values)
    )
    @. Pk.values = rho.values*min(
        Pk.values*nut.values,
        10*coeffs.β⁺*k.values*omega.values
    )

    correct_production!(Pk, boundaries.k, model, S.gradU, config) # Must be after Pk
    @. dkdomegadx.values = begin
        # 2*(F1.values - 1)*rho.values*coeffs.σω2*dkdomegadx.values/omega.values # explicit 
        2*(F1.values - 1)*rho.values*coeffs.σω2*dkdomegadx.values/omega.values/omega.values
    end
    

    # Solve omega equation
    # prev .= omega.values
    discretise!(ω_eqn, omega, config)
    apply_boundary_conditions!(ω_eqn, boundaries.omega, nothing, time, config)
    # implicit_relaxation!(ω_eqn, omega.values, solvers.omega.relax, nothing, config)
    implicit_relaxation_diagdom!(ω_eqn, omega.values, solvers.omega.relax, nothing, config)
    constrain_equation!(ω_eqn, boundaries.omega, model, config) # active with WFs only
    update_preconditioner!(ω_eqn.preconditioner, mesh, config)
    ω_res = solve_system!(ω_eqn, solvers.omega, omega, nothing, config)
    
    # constrain_boundary!(omega, omega.BCs, model, config) # active with WFs only
    bound!(omega, config)
    # explicit_relaxation!(omega, prev, solvers.omega.relax, config)

    # Solve k equation
    # prev .= k.values
    discretise!(k_eqn, k, config)
    apply_boundary_conditions!(k_eqn, boundaries.k, nothing, time, config)
    # implicit_relaxation!(k_eqn, k.values, solvers.k.relax, nothing, config)
    implicit_relaxation_diagdom!(k_eqn, k.values, solvers.k.relax, nothing, config)
    update_preconditioner!(k_eqn.preconditioner, mesh, config)
    k_res = solve_system!(k_eqn, solvers.k, k, nothing, config)
    bound!(k, config)
    # explicit_relaxation!(k, prev, solvers.k.relax, config)

    @. nut.values = coeffs.α1*k.values/max(
        coeffs.α1*omega.values, 
        # F2.values*sqrt(Ω.values)
        F2.values*Ω.values
        )

    interpolate!(nutf, nut, config)
    correct_boundaries!(nutf, nut, boundaries.nut, time, config)
    correct_eddy_viscosity!(nutf, boundaries.nut, model, config)

    state.residuals = ((:k , k_res),(:omega, ω_res))
    state.converged = k_res < solvers.k.convergence && ω_res < solvers.omega.convergence
    return nothing
end

# Specialise VTK writer
function save_output(model::Physics{T,F,SO,M,Tu,E,D,BI}, outputWriter, iteration, time, config
    ) where {T,F,SO,M,Tu<:KOmegaSST,E,D,BI}
    args = (
        ("U", model.momentum.U), 
        ("p", model.momentum.p),
        ("k", model.turbulence.k),
        ("omega", model.turbulence.omega),
        ("nut", model.turbulence.nut),
        ("y", model.turbulence.y),
    )
    write_results(iteration, time, model.domain, outputWriter, config.boundaries, args...)
end