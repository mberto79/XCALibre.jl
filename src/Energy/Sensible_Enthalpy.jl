export Sensible_Enthalpy
export Ttoh

# Model type definition
struct Sensible_Enthalpy{S1,S2,F1,F2,C} <: AbstractEnergyModel
    h::S1
    T::S2
    hf::F1
    Tf::F2
    coeffs::C
end
Adapt.@adapt_structure Sensible_Enthalpy

struct Sensible_Enthalpy_Model{E1}
    energy_eqn::E1 
end
Adapt.@adapt_structure Sensible_Enthalpy_Model

# Model API constructor
ENERGY{Sensible_Enthalpy}(; Tref = 288.15) = begin
    coeffs = (Tref=Tref, other=nothing)
    ARG = typeof(coeffs)
    ENERGY{Sensible_Enthalpy,ARG}(coeffs)
end

# Functor as consturctor
(energy::ENERGY{Sensible_Enthalpy, ARG})(mesh) where ARG = begin
    h = ScalarField(mesh)
    T = ScalarField(mesh)
    hf = FaceScalarField(mesh)
    Tf = FaceScalarField(mesh)
    coeffs = energy.args
    Sensible_Enthalpy(h, T, hf, Tf, coeffs)
end

function initialise(
    energy::Sensible_Enthalpy, model::Physics{T1,F,M,Tu,E,D,BI}, mdotf, peqn, config
    ) where {T1,F,M,Tu,E,D,BI}

    (; h, T) = energy
    (; solvers, schemes, runtime) = config
    mesh = mdotf.mesh
    eqn = peqn.equation
    
    rho = ScalarField(mesh)
    keff_by_cp = FaceScalarField(mesh)
    divK = ScalarField(mesh)

    h = Ttoh(model, T)

    energy_eqn = (
        Time{schemes.h.time}(rho, h)
        + Divergence{schemes.h.divergence}(mdotf, h) 
        - Laplacian{schemes.h.laplacian}(keff_by_cp, h) 
        == 
        -Source(divK)
    ) → eqn
    
    # Set up preconditioners
    @reset energy_eqn.preconditioner = set_preconditioner(
                solvers.h.preconditioner, energy_eqn, h.BCs, config)
    
    # preallocating solvers
    @reset energy_eqn.solver = solvers.h.solver(_A(energy_eqn), _b(energy_eqn))

    return Sensible_Enthalpy_Model(energy_eqn)
end

function energy!(
    energy::Sensible_Enthalpy_Model{E1}, model::Physics{T,F,M,Tu,E,D,BI}, prev, config
    ) where {T,F,M,Tu,E,D,BI,E1}

    mesh = model.domain

    (;U) = model.momentum
    (;h, hf) = model.energy
    (;energy_eqn) = energy
    (; solvers, runtime) = config

    mdotf = get_flux(energy_eqn, 2)
    keff_by_cp = get_flux(energy_eqn, 3)
    divK = get_source(energy_eqn, 1)

    Uf = FaceVectorField(mesh)
    Kf = FaceScalarField(mesh)
    K = ScalarField(mesh)

    interpolate!(Uf, U)
    correct_boundaries!(Uf, U, U.BCs)
    for i ∈ eachindex(K)
        K.values[i] = 0.5*(U.x.values[i]^2 + U.y.values[i]^2 + U.z.values[i]^2)
    end
    interpolate!(Kf, K)
    for i ∈ eachindex(Kf)
        Kf.values[i] = 0.5*(Uf.x.values[i]^2 + Uf.y.values[i]^2 + Uf.z.values[i]^2)
    end
    correct_face_interpolation!(Kf, K, mdotf)
    @. Kf.values *= mdotf.values
    divnovol!(divK, Kf)

    # Set up and solve energy equation
    @. prev = h.values
    discretise!(energy_eqn, prev, runtime)
    apply_boundary_conditions!(energy_eqn, h.BCs)
    implicit_relaxation_improved!(energy_eqn.equation, prev, solvers.energy.relax)
    update_preconditioner!(energy_eqn.preconditioner)
    run!(energy_eqn, solvers.energy)
    residual!(R_e, energy_eqn.equation, energy, iteration)
end

function thermo_Psi!(
    model::Physics{T,F,M,Tu,E,D,BI}, Psi::ScalarField, h::ScalarField
    ) where {T,F<:AbstractCompressible,M,Tu,E,D,BI}
    (; coeffs) = model.energy
    (; Tref) = coeffs
    Cp = _Cp(model.fluid); R = _R(model.fluid)
    @. Psi.values = Cp/(R*(h.values + Cp*Tref))
end

function thermo_Psi!(
    model::Physics{T,F,M,Tu,E,D,BI}, Psif::FaceScalarField, hf::FaceScalarField
    ) where {T,F<:AbstractCompressible,M,Tu,E,D,BI}
    (; coeffs) = model.energy
    (; Tref) = coeffs
    Cp = _Cp(model.fluid); R = _R(model.fluid)
    @. Psif.values = Cp/(R*(hf.values + Cp*Tref))
end


function thermo_rho!(h, t)

end

function Ttoh(
    model::Physics{T1,F,M,Tu,E,D,BI}, T::ScalarField
    ) where {T1,F<:AbstractCompressible,M,Tu,E,D,BI}
    (; coeffs) = model.energy
    (; Tref) = coeffs
    h = T
    Cp = _Cp(model.fluid)
    @. h.values = Cp.values*(T.values-Tref)
    return h
end

function Ttoh(
    model::Physics{T1,F,M,Tu,E,D,BI}, T
    ) where {T1,F<:AbstractCompressible,M,Tu,E,D,BI}
    (; coeffs) = model.energy
    (; Tref) = coeffs
    h = ScalarField(model.domain)
    Cp = _Cp(model.fluid)
    h = Cp.values*(T-Tref)
    return h
end

function htoT(
    model::Physics{T1,F,M,Tu,E,D,BI}, h::ScalarField
    ) where {T1,F<:AbstractCompressible,M,Tu,E,D,BI}
    (; coeffs) = model.energy
    (; Tref) = coeffs
    h = ScalarField(model.domain)
    Cp = _Cp(model.fluid)
    @. T.values = (h.values/Cp) + Tref
    return h
end



function clamp!(h, t)

end