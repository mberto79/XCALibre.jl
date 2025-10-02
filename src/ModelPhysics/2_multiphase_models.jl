export HelmholtzEnergy, HelmholtzEnergyFluid, H2, H2_para, N2

export Phase, Fluid, Multiphase
export ConstEos, PerfectGas, PengRobinson, ConstMu, Sutherland, Andrade
export Phase, physicsProperties

export ConstSurfaceTension, SurfaceTensionModel, NucleateBoilingModel, Drag_SchillerNaumann
# export LeeModel, DriftVelocity, Gravity, CSF, ArtificialCompression
export AbstractModel, AbstractEosModel, AbstractViscosityModel


abstract type AbstractModel end
abstract type AbstractEosModel <: AbstractModel end
abstract type AbstractViscosityModel <: AbstractModel end

abstract type AbstractPhase <: AbstractMultiphase end


Base.@kwdef struct ConstEos{T<:AbstractFloat} <: AbstractEosModel
    rho::T
end
(eos::ConstEos)(phase, model, config) = begin
    rho_field = phase.rho
    initialise!(rho_field, eos.rho)
end


Base.@kwdef struct PerfectGas{T<:AbstractFloat} <: AbstractEosModel
    rho::T
    R::T
end
(eos::PerfectGas)(phase, model, config) = begin
    (; p) = model.momentum

    T_ref = 273.0
    R = phase.eosModel.R
    rho_field = phase.rho

    @. rho_field.values = (p.values) / (R * T_ref) # CAREFUL WITH p=0 initialisation
end


Base.@kwdef struct PengRobinson{T<:AbstractFloat} <: AbstractEosModel
    T_crit::T
    p_crit::T
    omega::T
    M::T # Molar mass in g/mol
end
(eos::PengRobinson)(phase, model, config) = begin
    backend = config.hardware.backend
    workgroup = config.hardware.workgroup

    p = model.momentum.p

    if typeof(model.energy) <: Nothing # Isothermal
        T = ConstantScalar(273.0) # THIS PROBABLY NEEDS TO BE DEFINED BY USER! Redesign Isothermal Energy ?
    else
        T = model.energy.T
    end
    
    rho_l_field = model.fluid.phases[1].rho
    rho_p_field = model.fluid.phases[2].rho

    T_c = eos.T_crit
    p_c = eos.p_crit
    ω = eos.omega
    M = eos.M

    M = M * 1.0e-3

    ndrange = length(rho_l_field)
    kernel! = _peng_robinson(_setup(backend, workgroup, ndrange)...)
    kernel!(T, p, T_c, p_c, ω, M, rho_l_field, rho_p_field)
end


@kernel inbounds=true function _peng_robinson(T, p, T_c, p_c, ω, M, rho_l_field, rho_p_field)
    i = @index(Global)

    R_univ = 8.314

    T_r = T[i] / T_c

    a_H2 = 0.45724 * (((R_univ^2)*(T_c^2))/p_c)
    b_H2 = 0.07780 * ((R_univ*T_c)/p_c)

    κ = 0.37464 + 1.54226*ω - 0.26992*(ω^2)
    α = (1 + κ*(1 - sqrt(T_r)))^2
    
    A = ( a_H2 * α * p[i] ) / ( (R_univ^2) * (T[i]^2) )
    B = ( b_H2 * p[i])  / (R_univ * T[i])
    
    coeffs = [
        -(A*B - B^2 - B^3),   # constant term (c0)
        A - 2B - 3B^2,       # Z term (c1)
        -(1 - B),             # Z^2 term (c2)
        1.0                  # Z^3 term
    ]
    
    Z_roots = solve_cubic_eqn(coeffs[3], coeffs[2], coeffs[1]) # no need to take care of Z^3 because it is = 1.0

    # Extract only real roots
    Z_real = [z for z in Z_roots if isreal(z)]
    Z_real = real.(Z_real)                          # DO WE NEED THE DOT IN KERNEL ????

    Z_liq = minimum(Z_real)   # liquid root
    Z_vap = maximum(Z_real)   # vapour root

    # Convert to molar volumes
    Vm_liq = Z_liq * R_univ * T[i] / p[i]
    Vm_vap = Z_vap * R_univ * T[i] / p[i]

    # Convert to densities [kg/m3]
    rho_liq = M / Vm_liq
    rho_vap = M / Vm_vap

    rho_l_field[i] = rho_liq
    rho_p_field[i] = rho_vap
end

## COMMENT HERE
function solve_cubic_eqn(a::T, b::T, c::T) where {T<:AbstractFloat}
    p = b - a^2 / (T(3))
    q = (T(2) * a^3) / T(27) - (a * b) / T(3) + c

    Δ = (q / T(2))^2 + (p / T(3))^3

    if Δ > 0
        # One real root
        u = cbrt(-q / T(2) + sqrt(Δ))
        v = cbrt(-q / T(2) - sqrt(Δ))
        return (u + v) - a / T(3)
    else
        # Three real roots
        r = sqrt(-p / T(3))
        θ = acos(-q / (T(2) * r^3))
        twopi = T(2) * T(π)
        roots = (
            T(2) * r * cos(θ / T(3)) - a / T(3),
            T(2) * r * cos((θ + twopi / T(3))) - a / T(3),
            T(2) * r * cos((θ + T(2)*twopi / T(3))) - a / T(3),
        )
        return roots
    end
end



abstract type HelmholtzEnergyFluid end

struct N2 <: HelmholtzEnergyFluid end
struct H2 <: HelmholtzEnergyFluid end
struct H2_para <: HelmholtzEnergyFluid end

Base.@kwdef struct HelmholtzEnergy{F<:HelmholtzEnergyFluid}
    name::F
end





Base.@kwdef struct ConstMu{T<:AbstractFloat} <: AbstractViscosityModel
    mu::T
end
(viscosityModel::ConstMu)(phase, T) = begin
    mu_field = phase.mu
    initialise!(mu_field, viscosityModel.mu)
end


Base.@kwdef struct Sutherland{T<:AbstractFloat} <: AbstractViscosityModel
    mu_ref::T
    S::T
end
(viscosityModel::Sutherland)(phase, T) = begin
    mu_ref = viscosityModel.mu_ref
    S = viscosityModel.S

    T_ref = 273.0
    mu_field = phase.mu

    @. mu_field.values = (mu_ref * (T.values/T_ref)^1.5) * ((T_ref + S)/(T.values + S))
end


Base.@kwdef struct Andrade{T<:AbstractFloat} <: AbstractViscosityModel
    B::T
    C::T
end
(viscosityModel::Andrade)(phase, T) = begin
    B = viscosityModel.B
    C = viscosityModel.C
    
    mu_field = phase.mu

    @. mu_field.values = B * exp(C / T.values)
end


Base.@kwdef struct HydrogenViscosity <: AbstractPhysicsProperty end
Base.@kwdef struct NitrogenViscosity <: AbstractPhysicsProperty end



Base.@kwdef struct ConstSurfaceTension{T<:AbstractFloat} <: AbstractPhysicsProperty
    s::T
end


Base.@kwdef struct SurfaceTensionModel <: AbstractPhysicsProperty end
Base.@kwdef struct NucleateBoilingModel <: AbstractPhysicsProperty end



@kwdef struct Phase{E<:AbstractEosModel, V<:AbstractViscosityModel} <: AbstractPhase
    eosModel::E
    viscosityModel::V
end
@kwdef struct PhaseState{E<:AbstractEosModel, V<:AbstractViscosityModel, S1,S2,S3,S4,S5} <: AbstractPhase
    eosModel::E
    viscosityModel::V

    rho::S1
    mu::S2
    k::S3
    cp::S4
    beta::S5
end
Adapt.@adapt_structure PhaseState

function build_phase(phase_setup::Phase, mesh)
    rho   = ScalarField(mesh)
    mu    = ScalarField(mesh)
    k     = ScalarField(mesh)
    cp    = ScalarField(mesh)
    beta  = ScalarField(mesh)

    return PhaseState(
        eosModel=phase_setup.eosModel,
        viscosityModel=phase_setup.viscosityModel,
        rho=rho,
        mu=mu,
        k=k,
        cp=cp,
        beta=beta
    )
end




@kwdef struct Multiphase{P1,P2,S1,F1,S2,F2,S3,F3} <: AbstractMultiphase
    phases::P1
    physics_properties::P2
    alpha::S1
    alphaf::F1
    rho::S2
    rhof::F2
    nu::S3
    nuf::F3
end
Adapt.@adapt_structure Multiphase

Fluid{Multiphase}(; phases::Tuple, kwargs...) = begin
    coeffs = (; phases, kwargs...)
    ARG = typeof(coeffs)
    Fluid{Multiphase, ARG}(coeffs)
end


(fluid::Fluid{Multiphase, ARG})(mesh) where {ARG} = begin
    coeffs = fluid.args

    physics_properties = Base.structdiff(coeffs, (phases = nothing,))

    build_multiphase(coeffs.phases, physics_properties, mesh)
end


build_property(property, mesh) = property
build_property(setup::Gravity, mesh) = build_gravityModel(setup, mesh)
build_property(setup::LeeModel, mesh) = build_leeModel(setup, mesh)
build_property(setup::DriftVelocity, mesh) = build_driftVelocity(setup, mesh)
build_property(setup::CSF, mesh) = build_CSF_Model(setup, mesh)
build_property(setup::ArtificialCompression, mesh) = build_ArtificialCompressionModel(setup, mesh)

function build_multiphase(phase_setups::Tuple{<:AbstractPhase, <:AbstractPhase}, physics_properties_setup::NamedTuple, mesh)
    phases = map(setup -> build_phase(setup, mesh), phase_setups)

    built_properties = map(prop_setup -> build_property(prop_setup, mesh), physics_properties_setup)

    alpha  = ScalarField(mesh)
    alphaf = FaceScalarField(mesh)

    rho  = ScalarField(mesh)
    rhof = FaceScalarField(mesh)

    nu  = ScalarField(mesh)
    nuf = FaceScalarField(mesh)
    
    Multiphase(phases=phases, physics_properties=built_properties, alpha=alpha, alphaf=alphaf, rho=rho, rhof=rhof, nu=nu, nuf=nuf)
end