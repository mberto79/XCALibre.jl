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
    coeffs::C
end
Adapt.@adapt_structure Smagorinsky

struct SmagorinskyModel{T,D,S1}
    turbulence::T
    Δ::D 
    state::S1
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
    initialise(turbulence::Smagorinsky, model::Physics{T,F,M,Tu,E,D,BI}, mdotf, peqn
    ) where {T,F,M,Tu,E,D,BI}

Initialisation of turbulent transport equations.

### Input
- `turbulence`: turbulence model.
- `model`: Physics model defined by user.
- `mdtof`: Face mass flow.
- `peqn`: Pressure equation.

### Output
Returns a structure holding the fields and data needed for this model

    SmagorinskyModel(
        turbulence, 
        Δ, 
        ModelState((), false)
    )

"""
function initialise(
    turbulence::Smagorinsky, model::Physics{T,F,M,Tu,E,D,BI}, mdotf, peqn
    ) where {T,F,M,Tu,E,D,BI}

    (; solvers, schemes, runtime, boundaries) = get_configuration(CONFIG)
    mesh = model.domain
    
    Δ = ScalarField(mesh)

    delta!(Δ, mesh)
    (; coeffs) = model.turbulence
    @. Δ.values = (Δ.values*coeffs.C)^2.0
    
    return SmagorinskyModel(
        turbulence, 
        Δ, 
        ModelState((), false)
    )
end

# Model solver call (implementation)
"""
    turbulence!(les::SmagorinskyModel, model::Physics{T,F,M,Tu,E,D,BI}, S, S2, prev, time
    ) where {T,F,M,Tu<:AbstractTurbulenceModel,E,D,BI}

Run turbulence model transport equations.

### Input
- `les::SmagorinskyModel`: `Smagorinsky` LES turbulence model.
- `model`: Physics model defined by user.
- `S`: Strain rate tensor.
- `S2`: Square of the strain rate magnitude.
- `prev`: Previous field.
- `time`: current simulation time

"""
function turbulence!(
    les::SmagorinskyModel, model::Physics{T,F,M,Tu,E,D,BI}, S, prev, time
    ) where {T,F,M,Tu<:AbstractTurbulenceModel,E,D,BI}

    mesh = model.domain
    scalar = ScalarFloat(mesh)

    (; boundaries, hardware, schemes) = get_configuration(CONFIG)
    (; backend, workgroup) = hardware
    (; nut, nutf, coeffs) = les.turbulence
    (; U, Uf, gradU) = S
    (; Δ) = les

    
    grad!(gradU, Uf, U, boundaries.U, time) # update gradient (and S)
    limit_gradient!(schemes.U.limiter, gradU, U)
    
    wk = _setup(backend, workgroup, length(nut))[2] # index 2 to extract the workgroup
    AK.foreachindex(nut, min_elems=wk, block_size=wk) do i
        Si = S[i] # 0.5*(gradUi + gradUi')
        magS = sqrt(scalar(2.0)*Si⋅Si)
        nut[i] = Δ[i]*magS # Δ is (Cs*Δ)^2
    end

    interpolate!(nutf, nut)
    correct_boundaries!(nutf, nut, boundaries.nut, time)
    correct_eddy_viscosity!(nutf, boundaries.nut, model)
end

# Specialise VTK writer
function save_output(model::Physics{T,F,M,Tu,E,D,BI}, outputWriter, iteration, time
    ) where {T,F,M,Tu<:Smagorinsky,E,D,BI}
    args = (
        ("U", model.momentum.U), 
        ("p", model.momentum.p),
        ("nut", model.turbulence.nut)
    )
    config = get_configuration(CONFIG)
    write_results(iteration, time, model.domain, outputWriter, config.boundaries, args...)
end