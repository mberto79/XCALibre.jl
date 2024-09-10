export Smagorinsky

# Model type definition
"""
    Smagorinsky <: AbstractTurbulenceModel

Smagorinsky LES model containing all Smagorinksy field parameters.

### Fields
- 'nut' -- Eddy viscosity ScalarField.
- 'nutf' -- Eddy viscosity FaceScalarField.
- 'coeffs' -- Model coefficients.

"""
struct Smagorinsky{S1,S2,C} <: AbstractLESModel
    nut::S1
    nutf::S2
    coeff::C
end
Adapt.@adapt_structure Smagorinsky

struct SmagorinskyModel{D,S}
    Δ::D 
    magS::S
end
Adapt.@adapt_structure SmagorinskyModel

# Model API constructor (pass user input as keyword arguments and process as needed)
LES{Smagorinsky}(; C=0.15) = begin 
    coeff = (C=C,)
    ARG = typeof(coeff)
    LES{Smagorinsky,ARG}(coeff)
end

# Functor as constructor (internally called by Physics API): Returns fields and user data
(rans::LES{Smagorinsky, ARG})(mesh) where ARG = begin
    nut = ScalarField(mesh)
    nutf = FaceScalarField(mesh)
    coeff = rans.args
    Smagorinsky(nut, nutf, coeff)
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
- `SmagorinskyModel(Δ, magS)`  -- Turbulence model structure.

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
    
    return SmagorinskyModel(Δ, magS)
end

# Model solver call (implementation)
"""
    turbulence!(les::SmagorinskyModel{E1,E2}, model::Physics{T,F,M,Tu,E,D,BI}, S, S2, prev, time, config
    ) where {T,F,M,Tu<:Smagorinsky,E,D,BI,E1,E2}

Run turbulence model transport equations.

### Input
- `les::SmagorinskyModel{E1,E2}` -- Smagorinsky LES turbulence model.
- `model`  -- Physics model defined by user.
- `S`   -- Strain rate tensor.
- `S2`  -- Square of the strain rate magnitude.
- `prev`  -- Previous field.
- `time`   -- 
- `config` -- Configuration structure defined by user with solvers, schemes, runtime and 
              hardware structures set.

"""
function turbulence!(
    les::SmagorinskyModel{E1,E2}, model::Physics{T,F,M,Tu,E,D,BI}, S, S2, prev, time, config
    ) where {T,F,M,Tu<:Smagorinsky,E,D,BI,E1,E2}

    mesh = model.domain
    
    (; nut, nutf, coeff) = model.turbulence
    (; Δ, magS) = les

    magnitude!(magS, S, config)
    @. magS.values *= sqrt(2) # should fuse into definition of magnitude function!

    # update eddy viscosity 
    @. nut.values = coeff.C*Δ.values*magS.values # careful: here Δ = Δ²

    interpolate!(nutf, nut, config)
    correct_boundaries!(nutf, nut, nut.BCs, time, config)
    correct_eddy_viscosity!(nutf, nut.BCs, model, config)
end

# Specialise VTK writer
function model2vtk(model::Physics{T,F,M,Tu,E,D,BI}, VTKWriter, name
    ) where {T,F,M,Tu<:Smagorinsky,E,D,BI}
    args = (
        ("U", model.momentum.U), 
        ("p", model.momentum.p),
        ("nut", model.turbulence.nut)
    )
    write_vtk(name, model.domain, VTKWriter, args...)
end