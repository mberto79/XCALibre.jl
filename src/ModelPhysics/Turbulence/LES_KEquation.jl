export KEquation

# Model type definition
"""
    KEquation <: AbstractTurbulenceModel

KEquation LES model containing all Smagorinksy field parameters.

### Fields
- `nut` -- Eddy viscosity ScalarField.
- `nutf` -- Eddy viscosity FaceScalarField.
- `coeffs` -- Model coefficients.

"""
struct KEquation{S,SF,C} <: AbstractLESModel
    nut::S
    nutf::SF
    k::S
    kf::SF
    coeffs::C #I know there is only one coefficient for LES but this makes the DES implementation easier
end
Adapt.@adapt_structure KEquation

struct KEquationModel{T,D,S1,S2, E1}
    turbulence::T
    Δ::D 
    magS::S1
    k_eqn::E1
    state::S2
end
Adapt.@adapt_structure KEquationModel

# Model API constructor (pass user input as keyword arguments and process as needed)
LES{KEquation}(; ck=0.094, ce=1.048) = begin 
    coeffs = (ck=ck, ce=ce)
    ARG = typeof(coeffs)
    LES{KEquation,ARG}(coeffs)
end

# Functor as constructor (internally called by Physics API): Returns fields and user data
(les::LES{KEquation, ARG})(mesh) where ARG = begin
    nut = ScalarField(mesh)
    nutf = FaceScalarField(mesh)
    k = ScalarField(mesh)
    kf = FaceScalarField(mesh)
    coeffs = les.args
    KEquation(nut, nutf, k, kf, coeffs)
end

# Model initialisation
"""
    initialise(turbulence::KEquation, model::Physics{T,F,SO,M,Tu,E,D,BI}, mdotf, peqn, config
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
- `KEquationModel(
        turbulence, 
        Δ, 
        magS, 
        ModelState((), false)
    )`  -- Turbulence model structure.

"""
function initialise(
    turbulence::KEquation, model::Physics{T,F,SO,M,Tu,E,D,BI}, mdotf, peqn, config
    ) where {T,F,SO,M,Tu,E,D,BI}

    (; solvers, schemes, runtime) = config
    mesh = model.domain
    (; k, nut) = turbulence
    (; rho) = model.fluid
    eqn = peqn.equation
    
    magS = ScalarField(mesh)
    Δ = ScalarField(mesh)
    Pk = ScalarField(mesh)
    mueffk = FaceScalarField(mesh)
    Dkf = ScalarField(mesh)
    divU = ScalarField(mesh)
    kSource = ScalarField(mesh)

    delta!(Δ, mesh, config)

    k_eqn = (
            Time{schemes.k.time}(rho, k)
            + Divergence{schemes.k.divergence}(mdotf, k) 
            - Laplacian{schemes.k.laplacian}(mueffk, k) 
            + Si(Dkf,k) # Dkf = (Ce*rho*sqrt(k)/Δ)*k
            + Si(divU,k) # Needs adding
            ==
            Source(Pk)
        ) → eqn

    @reset k_eqn.preconditioner = set_preconditioner(solvers.k.preconditioner, k_eqn)
    @reset k_eqn.solver = _workspace(solvers.k.solver, _b(k_eqn))
    
    initial_residual = ((:k, 1.0),)
    return KEquationModel(
        turbulence, 
        Δ, 
        magS, 
        k_eqn,
        ModelState(initial_residual, false)
    ), config
end

# Model solver call (implementation)
"""
    turbulence!(les::KEquationModel, model::Physics{T,F,SO,M,Tu,E,D,BI}, S, S2, prev, time, config
    ) where {T,F,SO,M,Tu<:AbstractTurbulenceModel,E,D,BI}

Run turbulence model transport equations.

### Input
- `les::KEquationModel` -- KEquation LES turbulence model.
- `model`  -- Physics model defined by user.
- `S`   -- Strain rate tensor.
- `S2`  -- Square of the strain rate magnitude.
- `prev`  -- Previous field.
- `time`   -- 
- `config` -- Configuration structure defined by user with solvers, schemes, runtime and 
              hardware structures set.

"""
function turbulence!(
    les::KEquationModel, model::Physics{T,F,SO,M,Tu,E,D,BI}, S, prev, time, config
    ) where {T,F,SO,M,Tu<:AbstractTurbulenceModel,E,D,BI}

    mesh = model.domain
    (; solvers, runtime, hardware, boundaries) = config
    (; workgroup, backend) = hardware
    (; rho, rhof, nu, nuf) = model.fluid
    (; k, kf, nut, nutf, coeffs) = les.turbulence
    (; k_eqn, state) = les
    (; U, Uf, gradU) = S
    (; Δ, magS) = les

    Pk = get_source(k_eqn, 1)
    mdotf = get_flux(k_eqn, 2)
    mueffk = get_flux(k_eqn, 3)
    Dkf = get_flux(k_eqn, 4)
    divU = get_flux(k_eqn, 5)

    grad!(gradU, Uf, U, boundaries.U, time, config)
    limit_gradient!(config.schemes.U.limiter, gradU, U, config)
    
  
    wk = _setup(backend, workgroup, length(mueffk))[2]
    AK.foreachindex(mueffk, min_elems=wk, block_size=wk) do i 
        mueffk[i] = rhof[i]*(nuf[i] + nutf[i])
    end

    twoThirds = 2/3
    wk = _setup(backend, workgroup, length(Pk))[2]
    AK.foreachindex(Pk, min_elems=wk, block_size=wk) do i 
        Pk[i] = 2*nut[i]*rho[i]*(gradU[i]⋅Dev(S)[i])
        Dkf[i] = coeffs.ce*rho[i]*sqrt(k[i])/(Δ[i])
        divU[i] = twoThirds*rho[i]*tr(gradU[i])
    end
    
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

    wk = _setup(backend, workgroup, length(nut))[2]
    AK.foreachindex(nut, min_elems=wk, block_size=wk) do i 
        nut[i] = coeffs.ck*Δ[i]*sqrt(k[i])
    end

    interpolate!(nutf, nut, config)
    correct_boundaries!(nutf, nut, boundaries.nut, time, config)
    correct_eddy_viscosity!(nutf, boundaries.nut, model, config)

    # update solver state
    state.residuals = ((:k , k_res),)
    state.converged = k_res < solvers.k.convergence
    nothing
end

# Specialise VTK writer
function save_output(model::Physics{T,F,SO,M,Tu,E,D,BI}, outputWriter, iteration, time, config
    ) where {T,F,SO,M,Tu<:KEquation,E,D,BI}
    if typeof(model.fluid)<:AbstractCompressible
        args = (
            ("U", model.momentum.U), 
            ("p", model.momentum.p),
            ("k", model.turbulence.k),
            ("nut", model.turbulence.nut),
            ("T", model.energy.T)
        )
    else
        args = (
            ("U", model.momentum.U), 
            ("p", model.momentum.p),
            ("k", model.turbulence.k),
            ("nut", model.turbulence.nut)
        )
    end
    write_results(iteration, time, model.domain, outputWriter, config.boundaries, args...)
end