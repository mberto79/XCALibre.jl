export RAS, LES, Laminar
export Fluid, Incompressible, Compressible
export Simulation



# Simulation Simulation types 
struct Laminar end 

struct RAS{T,F1,F2,F,M}
    U::F1
    p::F2
    turbulence::F
    mesh::M
end

RAS{Laminar}(mesh) = begin
    U = VectorField(mesh)
    p = ScalarField(mesh)
    flag = false
    m = mesh
    RAS{Laminar,typeof(U),typeof(p),typeof(flag),typeof(m)}(U,p,flag,m)
end



# Simulation medium 
struct Fluid{T} end 

struct Incompressible end
struct Compressible end

struct Simulation{T,D,M,E}
    type
    domain
    medium
    energy
end

