export RANS
export dRANS
export Laminar
export AbstractMomentumModel, AbstractTurbulenceModel
export isturbulent

abstract type AbstractMomentumModel end
abstract type AbstractTurbulenceModel end

# Models 
struct Laminar <: AbstractMomentumModel end 

struct RANS{M,F1,F2,V,T,E,D}
    model::M
    U::F1 
    p::F2
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
    RANS{Laminar,F1,F2,V,F,F,D}(
        Laminar(), U, p, viscosity, flag, flag, mesh
    )
end

struct dRANS{M,F1,F2,F3,V,R,T,E,D}
    model::M
    U::F1
    h::F2 
    p::F3
    nu::V
    rho::R
    turbulence::T
    energy::E
    mesh::D
end 

dRANS{Laminar}(; mesh, viscosity) = begin
    U = VectorField(mesh); F1 = typeof(U)
    h = ScalarField(mesh); F2 = typeof(h)
    p = ScalarField(mesh); F3 = typeof(p)
    V = typeof(viscosity)
    rho = ScalarField(mesh); R = typeof(rho)
    flag = false; F = typeof(flag)
    D = typeof(mesh)
    dRANS{Laminar,F1,F2,F3,V,R,F,F,D}(
        Laminar(), U, h, p, viscosity, rho, flag, flag, mesh
    )
end



isturbulent(model) = begin
    typeof(model).parameters[1] <: AbstractTurbulenceModel
end