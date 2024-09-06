export Physics
export Transient, Steady

"""
    Physics

XCALibre's physcis model API.

### Fields
- 'time'   -- Time model.
- 'fluid'  -- Fluid model.
- 'momentum'  -- Momentum model.
- 'turbulence'  -- Turbulence model.
- 'energy'  -- Energy model.
- 'domain'  -- Mesh.
- 'boundary_info'  -- Mesh boundardy information.

### Examples
- `Phycsics(
    time::Union{Steady, Transient},
    fluid::AbstractFluid, 
    turbulence::AbstractTurbulenceModel,
    energy::AbstractEnergyModel,
    domain::AbstractMesh 
    )
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

"""
    Momentum

Momentum model containting key momentum fields.

### Fields
- 'U'        -- Velocity VectorField.
- 'p'        -- Pressure ScalarField.
- 'sources'  -- Momentum model sources.

### Examples
- `Momentum(mesh::AbstractMesh)
"""
struct Momentum{V,S,SS}
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