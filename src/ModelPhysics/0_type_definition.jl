export Physics
export AbstractMomentumModel
export Transient, Steady

"""
    struct Physics{T,F,M,Tu,E,D,BI}
        time::T
        fluid::F
        momentum::M 
        turbulence::Tu 
        energy::E
        domain::D
        boundary_info::BI
    end

XCALibre's parametric Physics type for user-level API. Also used to dispatch flow solvers.

### Fields
- 'time::Union{Steady, Transient}'   -- Time model.
- 'fluid::AbstractFluid'  -- Fluid model.
- 'momentum'  -- Momentum model.
- 'turbulence::AbstractTurbulenceModel'  -- Turbulence model.
- 'energy:AbstractEnergyModel'  -- Energy model.
- 'domain::AbstractMesh '  -- Mesh.
- 'boundary_info::boundary_info'  -- Mesh boundary information.

"""
struct Physics{T,F,M,Tu,E,D,BI}
    time::T
    fluid::F
    momentum::M 
    turbulence::Tu 
    energy::E
    domain::D
    boundary_info::BI
end 
Adapt.@adapt_structure Physics

"""
    Transient

Transient model for Physics model API.

### Examples
- `Transient()`
"""
struct Transient end
Adapt.@adapt_structure Transient

"""
    Steady

Steady model for Physics model API.

### Examples
- `Steady()`
"""
struct Steady end
Adapt.@adapt_structure Steady

abstract type AbstractMomentumModel end

"""
    struct Momentum{V,S,SS} <: AbstractMomentumModel
        U::V 
        p::S 
        sources::SS
    end 

Momentum model containing key momentum fields.

### Fields
- 'U'        -- Velocity VectorField.
- 'p'        -- Pressure ScalarField.
- 'sources'  -- Momentum model sources.

### Examples
- `Momentum(mesh::AbstractMesh)
"""
struct Momentum{V,S,SS} <: AbstractMomentumModel
    U::V 
    p::S 
    sources::SS
end 
Adapt.@adapt_structure Momentum 


Momentum(mesh::AbstractMesh) = begin
    U = VectorField(mesh)
    p = ScalarField(mesh)
    Momentum(U, p, nothing)
end

Physics(; time, fluid, turbulence, energy, domain) = begin
    momentum = Momentum(domain)
    fluid = fluid(domain)
    # turbulence = typeof(turbulence)(domain)
    turbulence = turbulence(domain)
    energy = energy(domain, fluid)
    boundary_info = boundary_map(domain)
    Physics(
        time,
        fluid,
        momentum, 
        turbulence, 
        energy,
        domain, 
        boundary_info
    )
end