export WALE

# Model type definition
"""
    WALE <: AbstractTurbulenceModel

Wall-Adapting Local Eddy-viscosity (WALE) LES model (Nicoud & Ducros, 1999) containing all
WALE field parameters. The eddy viscosity is computed algebraically as

    nut = (C*Δ)^2 * (Sd:Sd)^(3/2) / ((S:S)^(5/2) + (Sd:Sd)^(5/4))

where `S` is the strain rate tensor and `Sd` is the traceless symmetric part of the squared
velocity gradient tensor. The default model coefficient is `C = 0.325`.

### Fields
- `nut` -- Eddy viscosity ScalarField.
- `nutf` -- Eddy viscosity FaceScalarField.
- `coeffs` -- Model coefficients.

"""
struct WALE{S1,S2,C} <: AbstractLESModel
    nut::S1
    nutf::S2
    coeffs::C
end
Adapt.@adapt_structure WALE

struct WALEModel{T,D,S1,WS}
    turbulence::T
    Δ::D
    state::S1
    wall_scratch::WS
end
Adapt.@adapt_structure WALEModel

# Model API constructor (pass user input as keyword arguments and process as needed)
LES{WALE}(; C=0.325) = begin 
    coeffs = (C=C,)
    ARG = typeof(coeffs)
    LES{WALE,ARG}(coeffs)
end

# Functor as constructor (internally called by Physics API): Returns fields and user data
(les::LES{WALE, ARG})(mesh) where ARG = begin
    nut = ScalarField(mesh)
    nutf = FaceScalarField(mesh)
    coeffs = les.args
    WALE(nut, nutf, coeffs)
end

# Model initialisation
"""
    initialise(turbulence::WALE, model::Physics{T,F,SO,M,Tu,E,D,BI}, mdotf, peqn, config
    ) where {T,F,SO,M,Tu,E,D,BI}

Initialisation of turbulent transport equations.

### Input
- `turbulence`: turbulence model.
- `model`: Physics model defined by user.
- `mdtof`: Face mass flow.
- `peqn`: Pressure equation.
- `config`: Configuration structure defined by user with solvers, schemes, runtime and hardware structures set.

### Output
Returns a structure holding the fields and data needed for this model

    WALEModel(
        turbulence,
        Δ,
        ModelState((), false),
        wall_scratch
    )

"""
function initialise(
    turbulence::WALE, model::Physics{T,F,SO,M,Tu,E,D,BI}, mdotf, peqn, config
    ) where {T,F,SO,M,Tu,E,D,BI}

    (; solvers, schemes, runtime, boundaries) = config
    mesh = model.domain
    
    Δ = ScalarField(mesh)

    delta!(Δ, mesh, config)
    (; coeffs) = model.turbulence
    @. Δ.values = (Δ.values*coeffs.C)^2.0
    
    return WALEModel(
        turbulence, 
        Δ,
        ModelState((), false),
        wall_scratch(mesh, boundaries, config)
    ), config
end

# Model solver call (implementation)
"""
    turbulence!(les::WALEModel, model::Physics{T,F,SO,M,Tu,E,D,BI}, S, prev, time, config
    ) where {T,F,SO,M,Tu<:AbstractTurbulenceModel,E,D,BI}

Update the WALE eddy viscosity (algebraic model, no transport equations are solved).

### Input
- `les::WALEModel`: `WALE` LES turbulence model.
- `model`: Physics model defined by user.
- `S`: Strain rate tensor.
- `prev`: Previous field.
- `time`: current simulation time 
- `config`: Configuration structure defined by user with solvers, schemes, runtime and hardware structures set.

"""
function turbulence!(
    les::WALEModel, model::Physics{T,F,SO,M,Tu,E,D,BI}, S, prev, time, config
    ) where {T,F,SO,M,Tu<:AbstractTurbulenceModel,E,D,BI}

    mesh = model.domain

    (; boundaries, hardware) = config
    (; backend, workgroup) = hardware
    (; nut, nutf, coeffs) = les.turbulence
    (; U, Uf, gradU) = S
    (; Δ) = les

    
    grad!(gradU, Uf, U, boundaries.U, time, config) # update gradient (and S)
    limit_gradient!(config.schemes.U.limiter, gradU, U, config)
    
    wk = _setup(backend, workgroup, length(nut))[2] # index 2 to extract the workgroup
    AK.foreachindex(nut, min_elems=wk, block_size=wk) do i
        Si = S[i] # 0.5*(gradUi + gradUi')
        gradUi = gradU[i]
        SS = Si⋅Si # S:S

        # Traceless symmetric part of the squared velocity gradient tensor
        g2 = gradUi*gradUi
        Sd = 0.5*(g2 + g2') - (1/3)*tr(g2)*I
        SdSd = Sd⋅Sd # Sd:Sd

        num = SdSd^1.5
        den = SS^2.5 + SdSd^1.25
        nut[i] = Δ[i]*num/den # Δ is (Cw*Δ)^2
    end

    interpolate!(nutf, nut, config)
    correct_boundaries!(nutf, nut, boundaries.nut, time, config)
    correct_eddy_viscosity!(nutf, boundaries.nut, model, config, les.wall_scratch)
end

# Specialise VTK writer
function save_output(model::Physics{T,F,SO,M,Tu,E,D,BI}, outputWriter, iteration, time, config
    ) where {T,F,SO,M,Tu<:WALE,E,D,BI}
    if F <: Incompressible
        args = (
            ("U", model.momentum.U),
            ("p", model.momentum.p),
            ("nut", model.turbulence.nut)
        )
    elseif F <: SupersonicFlow
        args = (
            ("U", model.momentum.U),
            ("p", model.momentum.p),
            ("nut", model.turbulence.nut),
            ("T", model.energy.T),
            ("rho", model.fluid.rho)
        )
    else
        args = (
            ("U", model.momentum.U),
            ("p", model.momentum.p),
            ("nut", model.turbulence.nut),
            ("he", model.energy.T),
            ("rho", model.fluid.rho)
        )
    end
    write_results(iteration, time, model.domain, outputWriter, config.boundaries, args...)
end

