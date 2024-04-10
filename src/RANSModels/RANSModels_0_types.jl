export RANS
export Laminar
export AbstractMomentumModel, AbstractTurbulenceModel
export isturbulent

abstract type AbstractMomentumModel end
abstract type AbstractTurbulenceModel end

# Models 
struct Laminar <: AbstractMomentumModel end 

struct RANS{M,F1,F2,F3,F4,V,T,E,D}
    model::M
    U::F1 
    p::F2
    phi::F3
    y::F4
    nu::V
    turbulence::T
    energy::E
    mesh::D
end 

RANS{Laminar}(; mesh, viscosity) = begin
    U = VectorField(mesh); F1 = typeof(U)
    p = ScalarField(mesh); F2 = typeof(p)
    V = typeof(viscosity)
    flag = false; F = typeof(flag)
    D = typeof(mesh)
    RANS{Laminar,F1,F2,F,F,V,F,F,D}(
        Laminar(), U, p, flag, flag, viscosity, flag, flag, mesh
    )
end

isturbulent(model) = begin
    typeof(model).parameters[1] <: AbstractTurbulenceModel
end