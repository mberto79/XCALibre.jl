export KOmega

# Reference:
# Wilcox, D. C., Turbulence Modeling for CFD, 2nd edition, DCW Industries, Inc., La Canada CA, 1998

# Model type definition
"""
    KOmega <: AbstractTurbulenceModel

kOmega model containing all kOmega field parameters.

### Fields
- `k` -- Turbulent kinetic energy ScalarField.
- `omega` -- Specific dissipation rate ScalarField.
- `nut` -- Eddy viscosity ScalarField.
- `kf` -- Turbulent kinetic energy FaceScalarField.
- `omegaf` -- Specific dissipation rate FaceScalarField.
- `nutf` -- Eddy viscosity FaceScalarField.
- `coeffs` -- Model coefficients.

"""
struct KOmega{S1,S2,S3,F1,F2,F3,C} <: AbstractRANSModel
    k::S1
    omega::S2
    nut::S3
    kf::F1
    omegaf::F2
    nutf::F3
    coeffs::C
end
Adapt.@adapt_structure KOmega

struct KOmegaModel{T,E1,E2,S1} 
    turbulence::T
    k_eqn::E1 
    ω_eqn::E2
    state::S1
end
Adapt.@adapt_structure KOmegaModel

# Model API constructor (pass user input as keyword arguments and process as needed)
RANS{KOmega}(; β⁺=0.09, α1=0.52, β1=0.072, σk=0.5, σω=0.5) = begin 
    coeffs = (β⁺=β⁺, α1=α1, β1=β1, σk=σk, σω=σω)
    ARG = typeof(coeffs)
    RANS{KOmega,ARG}(coeffs)
end

# Functor as constructor (internally called by Physics API): Returns fields and user data
(rans::RANS{KOmega, ARG})(mesh) where ARG = begin
    k = ScalarField(mesh)
    omega = ScalarField(mesh)
    nut = ScalarField(mesh)
    kf = FaceScalarField(mesh)
    omegaf = FaceScalarField(mesh)
    nutf = FaceScalarField(mesh)
    coeffs = rans.args
    KOmega(k, omega, nut, kf, omegaf, nutf, coeffs)
end

# Model initialisation
"""
    initialise(turbulence::KOmega, model::Physics{T,F,M,Tu,E,D,BI}, mdotf, peqn, config
    ) where {T,F,M,Tu,E,D,BI}

Initialisation of turbulent transport equations.

### Input
- `turbulence` -- turbulence model.
- `model`  -- Physics model defined by user.
- `mdtof`  -- Face mass flow.
- `peqn`   -- Pressure equation.
- `config` -- Configuration structure defined by user with solvers, schemes, runtime and 
          hardware structures set.

### Output
- `KOmegaModel(
        turbulence,
        k_eqn, 
        ω_eqn,
        state
        )`  -- Turbulence model structure.

"""
function initialise(
    turbulence::KOmega, model::Physics{T,F,M,Tu,E,D,BI}, mdotf, peqn, config
    ) where {T,F,M,Tu,E,D,BI}

    (; k, omega, nut) = turbulence
    (; rho) = model.fluid
    (; solvers, schemes, runtime) = config
    mesh = mdotf.mesh
    eqn = peqn.equation

    # define fluxes and sources
    mueffk = FaceScalarField(mesh)
    mueffω = FaceScalarField(mesh)
    Dkf = ScalarField(mesh)
    Dωf = ScalarField(mesh)
    Pk = ScalarField(mesh)
    Pω = ScalarField(mesh)
    
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
            ==
            Source(Pω)
    ) → eqn

    # Set up preconditioners
    @reset k_eqn.preconditioner = set_preconditioner(
                solvers.k.preconditioner, k_eqn, k.BCs, config)

    # @reset ω_eqn.preconditioner = set_preconditioner(
    #             solvers.omega.preconditioner, ω_eqn, omega.BCs, config)

    @reset ω_eqn.preconditioner = k_eqn.preconditioner
    
    # preallocating solvers
    @reset k_eqn.solver = solvers.k.solver(_A(k_eqn), _b(k_eqn))
    @reset ω_eqn.solver = solvers.omega.solver(_A(ω_eqn), _b(ω_eqn))

    initial_residual = ((:k, 1.0),(:omega, 1.0))
    return KOmegaModel(turbulence, k_eqn, ω_eqn, ModelState(initial_residual, false))
end

# Model solver call (implementation)
"""
    turbulence!(rans::KOmegaModel, model::Physics{T,F,M,Tu,E,D,BI}, S, prev, time, config
    ) where {T,F,M,Tu<:AbstractTurbulenceModel,E,D,BI}

Run turbulence model transport equations.

### Input
- `rans::KOmegaModel` -- KOmega turbulence model.
- `model`  -- Physics model defined by user.
- `S`   -- Strain rate tensor.
- `prev`  -- Previous field.
- `time`   -- 
- `config` -- Configuration structure defined by user with solvers, schemes, runtime and 
              hardware structures set.

"""
function turbulence!(
    rans::KOmegaModel, model::Physics{T,F,M,Tu,E,D,BI}, S, prev, time, config
    ) where {T,F,M,Tu<:AbstractTurbulenceModel,E,D,BI}

    mesh = model.domain
    
    (; rho, rhof, nu, nuf) = model.fluid
    (;k, omega, nut, kf, omegaf, nutf, coeffs) = rans.turbulence
    (; U, Uf, gradU) = S
    (;k_eqn, ω_eqn, state) = rans
    (; solvers, runtime) = config

    mueffk = get_flux(k_eqn, 3)
    Dkf = get_flux(k_eqn, 4)
    Pk = get_source(k_eqn, 1)

    mueffω = get_flux(ω_eqn, 3)
    Dωf = get_flux(ω_eqn, 4)
    Pω = get_source(ω_eqn, 1)

    # update fluxes and sources

    # TO-DO: Need to bring gradient calculation inside turbulence models!!!!!

    grad!(gradU, Uf, U, U.BCs, time, config)
    limit_gradient!(config.schemes.U.limiter, gradU, U, config)
    magnitude2!(Pk, S, config, scale_factor=2.0) # multiplied by 2 (def of Sij)
    # constrain_boundary!(omega, omega.BCs, model, config) # active with WFs only
    
    @. Pω.values = rho.values*coeffs.α1*Pk.values
    @. Pk.values = rho.values*nut.values*Pk.values
    correct_production!(Pk, k.BCs, model, S.gradU, config) # Must be after previous line
    @. Dωf.values = rho.values*coeffs.β1*omega.values
    @. mueffω.values = rhof.values * (nuf.values + coeffs.σω*nutf.values)
    @. Dkf.values = rho.values*coeffs.β⁺*omega.values
    @. mueffk.values = rhof.values * (nuf.values + coeffs.σk*nutf.values)

    # Solve omega equation
    # prev .= omega.values
    discretise!(ω_eqn, omega, config)
    apply_boundary_conditions!(ω_eqn, omega.BCs, nothing, time, config)
    # implicit_relaxation!(ω_eqn, omega.values, solvers.omega.relax, nothing, config)
    implicit_relaxation_diagdom!(ω_eqn, omega.values, solvers.omega.relax, nothing, config)
    constrain_equation!(ω_eqn, omega.BCs, model, config) # active with WFs only
    update_preconditioner!(ω_eqn.preconditioner, mesh, config)
    ω_res = solve_system!(ω_eqn, solvers.omega, omega, nothing, config)
    
    # constrain_boundary!(omega, omega.BCs, model, config) # active with WFs only
    bound!(omega, config)
    # explicit_relaxation!(omega, prev, solvers.omega.relax, config)

    # Solve k equation
    # prev .= k.values
    discretise!(k_eqn, k, config)
    apply_boundary_conditions!(k_eqn, k.BCs, nothing, time, config)
    # implicit_relaxation!(k_eqn, k.values, solvers.k.relax, nothing, config)
    implicit_relaxation_diagdom!(k_eqn, k.values, solvers.k.relax, nothing, config)
    update_preconditioner!(k_eqn.preconditioner, mesh, config)
    k_res = solve_system!(k_eqn, solvers.k, k, nothing, config)
    bound!(k, config)
    # explicit_relaxation!(k, prev, solvers.k.relax, config)

    @. nut.values = k.values/omega.values

    interpolate!(nutf, nut, config)
    correct_boundaries!(nutf, nut, nut.BCs, time, config)
    correct_eddy_viscosity!(nutf, nut.BCs, model, config)

    state.residuals = ((:k , k_res),(:omega, ω_res))
    state.converged = k_res < solvers.k.convergence && ω_res < solvers.omega.convergence
    return nothing
end

# Specialise VTK writer
function model2vtk(model::Physics{T,F,M,Tu,E,D,BI}, VTKWriter, name
    ) where {T,F,M,Tu<:KOmega,E,D,BI}
    if typeof(model.fluid)<:AbstractCompressible
        args = (
            ("U", model.momentum.U), 
            ("p", model.momentum.p),
            ("T", model.energy.T),
            ("k", model.turbulence.k),
            ("omega", model.turbulence.omega),
            ("nut", model.turbulence.nut)
        )
    else
        args = (
            ("U", model.momentum.U), 
            ("p", model.momentum.p),
            ("k", model.turbulence.k),
            ("omega", model.turbulence.omega),
            ("nut", model.turbulence.nut)
        )
    end
    write_vtk(name, model.domain, VTKWriter, args...)
end