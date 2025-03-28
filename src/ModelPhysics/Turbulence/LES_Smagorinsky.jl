export Smagorinsky

# Model type definition
"""
    Smagorinsky <: AbstractTurbulenceModel

Smagorinsky LES model containing all Smagorinksy field parameters.

### Fields
- `nut` -- Eddy viscosity ScalarField.
- `nutf` -- Eddy viscosity FaceScalarField.
- `coeffs` -- Model coefficients.

"""
struct Smagorinsky{S1,S2,C} <: AbstractLESModel
    nut::S1
    nutf::S2
    coeffs::C #I know there is only one coefficient for LES but this makes the DES implementation easier
end
Adapt.@adapt_structure Smagorinsky

struct SmagorinskyModel{T,D,S1,S2}
    turbulence::T
    Δ::D 
    magS::S1
    state::S2
end
Adapt.@adapt_structure SmagorinskyModel

# Model API constructor (pass user input as keyword arguments and process as needed)
LES{Smagorinsky}(; C=0.15) = begin 
    coeffs = (C=C,)
    ARG = typeof(coeffs)
    LES{Smagorinsky,ARG}(coeffs)
end

# Functor as constructor (internally called by Physics API): Returns fields and user data
(les::LES{Smagorinsky, ARG})(mesh) where ARG = begin
    nut = ScalarField(mesh)
    nutf = FaceScalarField(mesh)
    coeffs = les.args
    Smagorinsky(nut, nutf, coeffs)
end

# Model initialisation
"""
    initialise(turbulence::Smagorinsky, model::Physics{T,F,M,Tu,E,D,BI}, mdotf, peqn, config
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
- `SmagorinskyModel(
        turbulence, 
        Δ, 
        magS, 
        ModelState((), false)
    )`  -- Turbulence model structure.

"""
function initialise(
    turbulence::Smagorinsky, model::Physics{T,F,M,Tu,E,D,BI}, mdotf, peqn, config
    ) where {T,F,M,Tu,E,D,BI}

    (; solvers, schemes, runtime) = config
    mesh = model.domain
    
    magS = ScalarField(mesh)
    Δ = ScalarField(mesh)

    delta!(Δ, mesh, config)
    @. Δ.values = Δ.values^2 # store delta squared since it will be needed
    
    return SmagorinskyModel(
        turbulence, 
        Δ, 
        magS, 
        ModelState((), false)
    )
end

# Model solver call (implementation)
"""
    turbulence!(les::SmagorinskyModel, model::Physics{T,F,M,Tu,E,D,BI}, S, S2, prev, time, config
    ) where {T,F,M,Tu<:AbstractTurbulenceModel,E,D,BI}

Run turbulence model transport equations.

### Input
- `les::SmagorinskyModel` -- Smagorinsky LES turbulence model.
- `model`  -- Physics model defined by user.
- `S`   -- Strain rate tensor.
- `S2`  -- Square of the strain rate magnitude.
- `prev`  -- Previous field.
- `time`   -- 
- `config` -- Configuration structure defined by user with solvers, schemes, runtime and 
              hardware structures set.

"""
function turbulence!(
    les::SmagorinskyModel, model::Physics{T,F,M,Tu,E,D,BI}, S, prev, time, config
    ) where {T,F,M,Tu<:AbstractTurbulenceModel,E,D,BI}

    mesh = model.domain
    
    (; nut, nutf, coeffs) = les.turbulence
    (; U, Uf, gradU) = S
    (; Δ, magS) = les

    grad!(gradU, Uf, U, U.BCs, time, config) # update gradient (internal structure of S)
    limit_gradient!(config.schemes.U.limiter, gradU, U, config)
    magnitude!(magS, S, config)
    @. magS.values *= sqrt(2) # should fuse into definition of magnitude function!

    # update eddy viscosity 
    @. nut.values = coeffs.C*Δ.values*magS.values # careful: here Δ = Δ²

    interpolate!(nutf, nut, config)
    correct_boundaries!(nutf, nut, nut.BCs, time, config)
    correct_eddy_viscosity!(nutf, nut.BCs, model, config)
end

# Specialise VTK writer
function save_output(model::Physics{T,F,M,Tu,E,D,BI}, outputWriter, iteration
    ) where {T,F,M,Tu<:Smagorinsky,E,D,BI}
    args = (
        ("U", model.momentum.U), 
        ("p", model.momentum.p),
        ("nut", model.turbulence.nut)
    )
    write_results(iteration, model.domain, outputWriter, args...)
end