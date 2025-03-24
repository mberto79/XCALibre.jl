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
struct KOmegaSST{S1,S2,S3,F1,F2,F3,C1,C2,C3,Y} <: AbstractRANSModel
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
end
Adapt.@adapt_structure KOmegaSST

struct KOmegaSSTModel{E1,E2,S,S1,S2,S3,S4,S5,S6,S7,S8,S9,S10}
    k_eqn::E1 
    ω_eqn::E2
    state::S
    β::S1
    σk::S2
    σω::S3
    γ::S4  
    CDkω::S5
    arg1::S6
    F1::S7
    arg2::S8
    F2::S9
    Ω::S10
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
    walls = rans.args.walls
    BCs = []
    for boundary ∈ mesh.boundaries
        for namedwall ∈ walls
            if boundary.name == namedwall
                push!(BCs, Dirichlet(boundary.name, 0.0))
            else
                push!(BCs, Neumann(boundary.name, 0.0))
            end
        end
    end
    y = assign(y, BCs...)

    KOmegaSST(k, omega, nut, kf, omegaf, nutf, coeffs, gamma1, gamma2, y) 
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
- `KOmegaModel(k_eqn, ω_eqn)`  -- Turbulence model structure.

"""
function initialise(
    turbulence::KOmegaSST, model::Physics{T,F,M,Tu,E,D,BI}, mdotf, peqn, config
    ) where {T,F,M,Tu,E,D,BI}

    (; k, omega, nut, y) = turbulence
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
    CrossDiff = ScalarField(mesh)

    CDkω = ScalarField(mesh)
    arg1 = ScalarField(mesh)
    F1 = ScalarField(mesh)

    arg2 = ScalarField(mesh)
    F2 = ScalarField(mesh)

    β = ScalarField(mesh)
    σk = ScalarField(mesh)
    σω = ScalarField(mesh)
    γ = ScalarField(mesh)

    Ω = ScalarField(mesh)
    
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
            Source(Pω) + Source(CrossDiff)
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

    wall_distance!(model, config)

    initial_residual = ((:k, 1.0),(:omega, 1.0))
    return KOmegaSSTModel(k_eqn, ω_eqn, ModelState(initial_residual, false), β, σk, σω, γ, CDkω, arg1, F1, arg2, F2, Ω)
end

# Model solver call (implementation)
"""
    turbulence!(rans::KOmegaModel{E1,E2,S1}, model::Physics{T,F,M,Tu,E,D,BI}, S, prev, time, config
    ) where {T,F,M,Tu<:KOmega,E,D,BI,E1,E2,S1}

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
    rans::KOmegaSSTModel{E1,E2,S1}, model::Physics{T,F,M,Tu,E,D,BI}, S, prev, time, config
    ) where {T,F,M,Tu<:AbstractTurbulenceModel,E,D,BI,E1,E2,S1}

    mesh = model.domain
    
    (; rho, rhof, nu, nuf) = model.fluid
    (;k, omega, nut, kf, omegaf, nutf, coeffs, gamma1, gamma2, y) = model.turbulence
    (; U, Uf, gradU) = S
    (;k_eqn, ω_eqn, state, β, σk, σω, γ, CDkω, arg1, F1, arg2, F2, Ω) = rans
    (; solvers, runtime) = config

    mueffk = get_flux(k_eqn, 3)
    Dkf = get_flux(k_eqn, 4)
    Pk = get_source(k_eqn, 1)

    mueffω = get_flux(ω_eqn, 3)
    Dωf = get_flux(ω_eqn, 4)
    Pω = get_source(ω_eqn, 1)
    CrossDiff = get_source(ω_eqn, 2)

    # update fluxes and sources

    # TO-DO: Need to bring gradient calculation inside turbulence models!!!!!

    grad!(gradU, Uf, U, U.BCs, time, config)
    limit_gradient!(config.schemes.U.limiter, gradU, U, config)
    magnitude2!(Pk, S, config, scale_factor=2.0) # multiplied by 2 (def of Sij)
    # constrain_boundary!(omega, omega.BCs, model, config) # active with WFs only

    grad!(gradk, kf, k, k.BCs, time, config)
    grad!(gradω , omegaf, omega, omega.BCs, time, config)
     
    @. CDkω = max.(2*rho.values*coeffs.σω2*(1.0/omega.values)*(gradk.values.x*gradω.values.x + gradk.values.y*gradω.values.y +gradk.values.z*gradω.values.z), 1e-20)
    @. arg1 = min.(max.(sqrt(k.values)/(betastar*omega.values*y.values), 500.0*nu.values/(y.values*y.values*omega.values), 4.0*rho.values*coeffs.σω2*k.values/(CDkω*y.values*y.values)))
    @. F1 = tanh(arg1.values^4)

    @. arg2 = max.(2*sqrt(k.values)/(betastar*omega.values*y.values), 500.0*nu.values/(y.values*y.values*omega.values))
    @. F2 = tanh(arg2.values*arg2.values)

    @. σk = coeffs.σk1*F1 + (1.0 - F1)*coeffs.σk2
    @. σω = coeffs.σω1*F1 + (1.0 - F1)*coeffs.σω2
    @. σω = coeffs.β1*F1 + (1.0 - F1)*coeffs.β2
    @. γ = gamma1*F1 + (1.0 - F1)*gamma2


    @. Pω.values = rho.values*γ*Pk.values
    @. Pk.values = rho.values*nut.values*Pk.values
    @. CrossDiff.values = 2.0*(1.0-F1.values)*rho.values*coeffs.σω2*(1.0/omega.values)*(gradk.values.x*gradω.values.x + gradk.values.y*gradω.values.y +gradk.values.z*gradω.values.z)
    correct_production!(Pk, k.BCs, model, S.gradU, config) # Must be after previous line
    @. Dωf.values = rho.values*β.values*omega.values
    @. mueffω.values = rhof.values * (nuf.values + σω.values*nutf.values)
    @. Dkf.values = rho.values*coeffs.β⁺*omega.values
    @. mueffk.values = rhof.values * (nuf.values + σk.values*nutf.values)

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

    magnitude2!(Ω, S, config, scale_factor=2.0) # 
    @. nut.values = coeffs.a1*k.values/(max.(coeffs.a1*omega.values, F2.values*Ω.values))

    interpolate!(nutf, nut, config)
    correct_boundaries!(nutf, nut, nut.BCs, time, config)
    correct_eddy_viscosity!(nutf, nut.BCs, model, config)

    state.residuals = ((:k , k_res),(:omega, ω_res))
    state.converged = k_res < solvers.k.convergence && ω_res < solvers.omega.convergence
    return nothing
end

# Specialise VTK writer
function model2vtk(model::Physics{T,F,M,Tu,E,D,BI}, VTKWriter, name
    ) where {T,F,M,Tu<:KOmegaSST,E,D,BI}
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