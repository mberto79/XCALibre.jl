export ConstantRho
export thermo_Psi!
export thermo_rho!

struct ConstantRho <: AbstractThermoModel end

# Constrcutor 
ThermoModel{ConstantRho}(; mesh, rho_constant = 1.0) = begin
    flag = false
    rho = ConstantScalar(rho_constant); F1 = typeof(rho)
    Psi = ConstantScalar(rho_constant); F2 = typeof(Psi)
    rhof = ConstantScalar(rho_constant); F3 = typeof(rhof)
    Psif = ConstantScalar(rho_constant); F4 = typeof(Psif)
    fluidprop = flag; P = typeof(fluidprop)
    ThermoModel{ConstantRho,F1,F2,F3,F4,P}(
        ConstantRho(), rho, Psi, rhof, Psif, fluidprop
    )
end

function thermo_Psi!(thermodel::ConstantRho, h, hf, Psi, Psif)

end

function thermo_rho!(thermodel::ConstantRho, p, pf, Psi, Psif, rho, rhof)
    (;rho, rhof) = thermodel

end