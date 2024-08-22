export Physics
export Transient, Steady


struct Physics{T,F,M,Tu,E,D,BI,P}
    time::T
    fluid::F
    momentum::M 
    turbulence::Tu 
    energy::E
    domain::D
    boundary_info::BI
    periodic::P
end 
Adapt.@adapt_structure Physics

struct Transient end
Adapt.@adapt_structure Transient

struct Steady end
Adapt.@adapt_structure Steady

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

Physics(; time, fluid, turbulence, energy, domain, periodic=nothing) = begin
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
        boundary_info,
        periodic
    )
end