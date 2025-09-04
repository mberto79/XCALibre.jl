using XCALibre

# Current logic works in a way that when we are outside of saturation region - solver returns one physical density,
#       and the other one is a reference at saturation. Lee model is required to bring it back to physical state.


_eos_wrapper(fluid::H2, T, P) = XCALibre.ModelPhysics.EOS_wrapper_H2(T, P)
_eos_wrapper(fluid::N2, T, P) = XCALibre.ModelPhysics.EOS_wrapper_N2(T, P)

_mu_high_fidelity(fluid::H2, T, rho) = XCALibre.ModelPhysics.mu_high_fidelity_H2(T, rho)
_mu_high_fidelity(fluid::N2, T, rho) = XCALibre.ModelPhysics.mu_high_fidelity_N2(T, rho)

_thermal_conductivity(fluid::H2, args...) = XCALibre.ModelPhysics.thermal_conductivity_H2(args...)
_thermal_conductivity(fluid::N2, args...) = XCALibre.ModelPhysics.thermal_conductivity_N2(args...)


# This function is called per cell by kernel
(eos::HelmholtzEnergy)(T_input, P_input) = begin # beta0 value is not tested
    rho0, cv0, cp0, kT0, kT_ref, internal_energy0, 
            enthalpy0, entropy0, beta0, latentHeat0, T_sat0 = _eos_wrapper(eos.name, T_input, P_input) #m_qp, m_pq
        
    nu_bar_vals = [0.0, 0.0]
    k0_vals = [0.0, 0.0]

    for i in eachindex(rho0)
        nu_bar_vals[i] = _mu_high_fidelity(eos.name, T_input, rho0[i])
        k0_vals[i] = _thermal_conductivity(eos.name, rho0[i], T_input, cp0[i], cv0[i], kT0[i], kT_ref[i], nu_bar_vals[i])
    end

    surface_tension = calculate_surface_tension(eos.name, T_input)

    @. nu_bar_vals = nu_bar_vals * 1.0e-6 # convert to SI

    return (rho=rho0, cv=cv0, cp=cp0, u=internal_energy0, h=enthalpy0, s=entropy0, beta=beta0,
            mu=nu_bar_vals, k=k0_vals, sigma=surface_tension, L_vap=latentHeat0, T_sat=T_sat0) #m_lv=m_lv, m_vl=m_vl
end


