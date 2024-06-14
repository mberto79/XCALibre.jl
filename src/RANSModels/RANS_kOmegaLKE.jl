export KOmegaLKE

# Model type definition (hold fields)
struct KOmegaLKE{S1,S2,S3,S4,F1,F2,F3,F4,C} <: AbstractTurbulenceModel 
    k::S1
    omega::S2
    kl::S3
    nut::S4
    kf::F1
    omegaf::F2
    klf::F3
    nutf::F4
    coeffs::C
end 
Adapt.@adapt_structure KOmegaLKE

# Model type definition (hold equation definitions and data)
struct KOmegaLKEModel{E1,E2,E3,E4} <: AbstractTurbulenceModel
    k_eqn::E1
    omega_eqn::E1
    kl_eqn::E1
end 
Adapt.@adapt_structure KOmegaLKEModel

# Model API constructor
RANS{KOmegaLKE}(mesh) = begin
    k = ScalarField(mesh)
    omega = ScalarField(mesh)
    kl = ScalarField(mesh)
    nut = ScalarField(mesh)
    kf = FaceScalarField(mesh)
    omegaf = FaceScalarField(mesh)
    klf = FaceScalarField(mesh)
    nutf = FaceScalarField(mesh)
    coeffs = (
        C1 = 0.02974,
        C2 = 59.79,
        C3 = 1.191,
        C4 = 1.65*10^-13,
        Cμ = 0.09,
        Cω1 = 0.52,
        Cω2 = 0.0708,
        Ccrit = 76500,
        CSS = 1.45,
        Cv = 0.43,
        σk = 0.5,
        σd = 0.125,
        σkL = 0.0125,
        σω = 0.5
    )
    KOmegaLKE(k, omega, kl, nut, kf, omegaf, klf, nutf, coeffs)
end

# Model initialisation
function initialise(
    turbulence::KOmegaLKE, model::Physics{T,F,M,Tu,E,D,BI}, mdotf, peqn, config
    ) where {T,F,M,Tu,E,D,BI}
    return KOmegaLKEModel()
end

# Model solver call (implementation)
function turbulence!(rans::KOmegaLKEModel, model::Physics{T,F,M,Tu,E,D,BI}, S, S2, prev, config
    ) where {T,F,M,Tu<:KOmegaLKE,E,D,BI}
    nothing
end

# Specialise VTK writer
function model2vtk(model::Physics{T,F,M,Tu,E,D,BI}, name) where {T,F,M,Tu<:KOmegaLKE,E,D,BI}
    args = (
        ("U", model.momentum.U), 
        ("p", model.momentum.p)
    )
    write_vtk(name, model.domain, args...)
end