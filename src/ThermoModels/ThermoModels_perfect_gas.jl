export IdealGas
export thermo_Psi!
export thermo_rho!
export thermo_clamp!

# struct IdealGas <: AbstractThermoModel end

struct IdealGas
    gamma::Float64
    Cp::Float64
    Tref::Float64
end

function thermo_Psi!(thermodel::IdealGas, h::ScalarField, hf::FaceScalarField, Psi::ScalarField, Psif::FaceScalarField)
    (;gamma, Cp, Tref) = thermodel
    R = Cp*(1.0-1.0/gamma)
    @. Psi.values = Cp/(R*(h.values + Cp*Tref))
    @. Psif.values = Cp/(R*(hf.values + Cp*Tref))
end

function thermo_rho!(thermodel::IdealGas, p, pf, Psi, Psif, rho, rhof)
    @. rho.values = Psi.values*p.values
    @. rhof.values = Psif.values*pf.values
end

function thermo_clamp!(thermodel::IdealGas, h::ScalarField, Tmin, Tmax)
    (;gamma, Cp, Tref) = thermodel
    hmin = Cp*(Tmin-Tref); hmax = Cp*(Tmax-Tref)
    clamp!(h.values, hmin, hmax)
end