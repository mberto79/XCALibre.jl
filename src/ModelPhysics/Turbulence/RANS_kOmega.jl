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
- `nutf` -- Eddy viscosity FaceScalarField.
- `coeffs` -- Model coefficients.

"""
struct KOmega{S1,S2,S3,F3,C} <: AbstractRANSModel
    k::S1
    omega::S2
    nut::S3
    nutf::F3
    coeffs::C
end
Adapt.@adapt_structure KOmega

struct KOmegaModel{T,E1,E2,S1,WS} 
    turbulence::T
    k_eqn::E1 
    ω_eqn::E2
    state::S1
    wall_scratch::WS
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
    nutf = FaceScalarField(mesh)
    scalar = ScalarFloat(mesh)
    coeffs = (
        β⁺=scalar(rans.args.β⁺),
        α1=scalar(rans.args.α1),
        β1=scalar(rans.args.β1),
        σk=scalar(rans.args.σk),
        σω=scalar(rans.args.σω),
    )
    KOmega(k, omega, nut, nutf, coeffs)
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
- `KOmegaModel(
        turbulence,
        k_eqn, 
        ω_eqn,
        state
        )`  -- Turbulence model structure.

"""
function initialise(
    turbulence::KOmega, model::Physics{T,F,SO,M,Tu,E,D,BI}, mdotf, peqn, config
    ) where {T,F,SO,M,Tu,E,D,BI}

    (; k, omega, nut) = turbulence
    (; rho) = model.fluid
    (; solvers, schemes, runtime, boundaries) = config
    mesh = model.domain
    eqn = peqn.equation

    # define fluxes and sources
    mueffk = FaceScalarField(mesh, store_mesh=false)
    mueffω = FaceScalarField(mesh, store_mesh=false)
    Dkf = ScalarField(mesh, store_mesh=false)
    Dωf = ScalarField(mesh, store_mesh=false)
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

    # Krylov preconditioner/workspace are serial-only (distributed solves through PETSc PCs)
    if !is_distributed_mesh(mesh)
        @reset k_eqn.preconditioner = set_preconditioner(solvers.k.preconditioner, k_eqn)
        @reset ω_eqn.preconditioner = k_eqn.preconditioner
        @reset k_eqn.solver = _workspace(solvers.k.solver, _b(k_eqn))
        @reset ω_eqn.solver = _workspace(solvers.omega.solver, _b(ω_eqn))
    end

    # wrap transported-scalar eqns for the distributed solve seam (identity serial). This is
    # the single hook that makes any turbulence model distributed-capable.
    k_eqn = wrap_eqn(k_eqn, mesh, solvers.k, config; label="k")
    ω_eqn = wrap_eqn(ω_eqn, mesh, solvers.omega, config; label="omega")

    initial_residual = ((:k, 1.0),(:omega, 1.0))
    return KOmegaModel(
        turbulence, k_eqn, ω_eqn, ModelState(initial_residual, false),
        wall_scratch(mesh, boundaries, config)
        ), config
end

# Model solver call (implementation)
"""
    turbulence!(rans::KOmegaModel, model::Physics{T,F,SO,M,Tu,E,D,BI}, S, prev, time, config
    ) where {T,F,SO,M,Tu<:AbstractTurbulenceModel,E,D,BI}

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
    rans::KOmegaModel, model::Physics{T,F,SO,M,Tu,E,D,BI}, S, prev, time, config
    ) where {T,F,SO,M,Tu<:AbstractTurbulenceModel,E,D,BI}

    mesh = model.domain
    distributed = is_distributed_mesh(mesh)

    (; rho, rhof, nu, nuf) = model.fluid
    (;k, omega, nut, nutf, coeffs) = rans.turbulence
    (; U, Uf, gradU) = S
    (;k_eqn, ω_eqn, state, wall_scratch) = rans
    (; solvers, runtime, boundaries) = config

    # wrapped eqns solve through the seam; raw eqns are assembled/discretised in place
    k_deqn, ω_deqn = k_eqn, ω_eqn
    k_eqn, ω_eqn = unwrap_eqn(k_eqn), unwrap_eqn(ω_eqn)

    mueffk = get_flux(k_eqn, 3)
    Dkf = get_flux(k_eqn, 4)
    Pk = get_source(k_eqn, 1)

    mueffω = get_flux(ω_eqn, 3)
    Dωf = get_flux(ω_eqn, 4)
    Pω = get_source(ω_eqn, 1)

    # update fluxes and sources

    # TO-DO: Need to bring gradient calculation inside turbulence models!!!!!

    grad!(gradU, Uf, U, boundaries.U, time, config)
    limit_gradient!(config.schemes.U.limiter, gradU, U, config)
    # One pass over the cells and one over the faces: the strain-rate magnitude feeds both
    # productions, so writing it to Pk and reading it back twice was three passes for one.
    # Every field is bound with `field_values` so the closures carry values, not meshes.
    gradUv = field_values(gradU.result)
    rhov, omegav, nutv = field_values(rho), field_values(omega), field_values(nut)
    Pkv, Pωv, Dkv, Dωv = field_values(Pk), field_values(Pω), field_values(Dkf), field_values(Dωf)
    xcal_foreach(Pkv, config) do i
        @inbounds begin
            gradi = gradUv[i]
            Sij = 0.5*(gradi + gradi')
            GbyNu = 2*sum(Sij .* Sij)
            rhoi = rhov[i]
            omegai = omegav[i]
            Pωv[i] = rhoi*coeffs.α1*GbyNu
            Pkv[i] = rhoi*nutv[i]*GbyNu
            Dωv[i] = rhoi*coeffs.β1*omegai
            Dkv[i] = rhoi*coeffs.β⁺*omegai
        end
    end
    correct_production!(Pk, boundaries.k, model, S.gradU, config, wall_scratch) # Must be after previous line
    rhofv, nufv, nutfv = field_values(rhof), field_values(nuf), field_values(nutf)
    mueffkv, mueffωv = field_values(mueffk), field_values(mueffω)
    xcal_foreach(mueffkv, config) do i
        @inbounds begin
            rhofi = rhofv[i]
            nufi = nufv[i]
            nutfi = nutfv[i]
            mueffωv[i] = rhofi*(nufi + coeffs.σω*nutfi)
            mueffkv[i] = rhofi*(nufi + coeffs.σk*nutfi)
        end
    end

    # Solve omega equation
    # prev .= omega.values
    discretise!(ω_eqn, omega, config)
    apply_boundary_conditions!(ω_eqn, boundaries.omega, nothing, time, config)
    # implicit_relaxation!(ω_eqn, omega.values, solvers.omega.relax, nothing, config)
    implicit_relaxation_diagdom!(ω_eqn, omega.values, solvers.omega.relax, nothing, config)
    constrain_equation!(ω_eqn, boundaries.omega, model, config, wall_scratch) # active with WFs only
    distributed || update_preconditioner!(ω_eqn.preconditioner, mesh, config)
    ω_res = solve_system!(ω_deqn, solvers.omega, omega, nothing, config)

    # constrain_boundary!(omega, boundaries.omega, model, config) # active with WFs only
    bound!(omega, config)
    # explicit_relaxation!(omega, prev, solvers.omega.relax, config)

    # Solve k equation
    # prev .= k.values
    discretise!(k_eqn, k, config)
    apply_boundary_conditions!(k_eqn, boundaries.k, nothing, time, config)
    # implicit_relaxation!(k_eqn, k.values, solvers.k.relax, nothing, config)
    implicit_relaxation_diagdom!(k_eqn, k.values, solvers.k.relax, nothing, config)
    distributed || update_preconditioner!(k_eqn.preconditioner, mesh, config)
    k_res = solve_system!(k_deqn, solvers.k, k, nothing, config)
    bound!(k, config)
    # explicit_relaxation!(k, prev, solvers.k.relax, config)

    @. nut.values = k.values/omega.values

    interpolate!(nutf, nut, config)
    correct_boundaries!(nutf, nut, boundaries.nut, time, config)
    correct_eddy_viscosity!(nutf, boundaries.nut, model, config, wall_scratch)

    state.residuals = ((:k , k_res),(:omega, ω_res))
    state.converged = k_res < solvers.k.convergence && ω_res < solvers.omega.convergence
    return nothing
end

# Specialise VTK writer
function save_output(model::Physics{T,F,SO,M,Tu,E,D,BI}, outputWriter, iteration, time, config
    ) where {T,F,SO,M,Tu<:KOmega,E,D,BI}
    if typeof(model.fluid)<:AbstractCompressible
        args = (
            ("U", model.momentum.U), 
            ("p", model.momentum.p),
            ("rho", model.fluid.rho),
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
    write_results(iteration, time, model.domain, outputWriter, config.boundaries, args...)
end
