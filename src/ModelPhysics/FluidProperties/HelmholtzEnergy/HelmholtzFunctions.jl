export HelmholtzFluidConstants,
       find_density_advanced,
       params_computation,
       find_saturation_properties,
       EOS_wrapper


"""
Stores all fluid-specific constants required for the Helmholtz Equation of State,
ancillary equations, and associated property calculations.
"""
struct HelmholtzFluidConstants{T<:AbstractFloat,V<:AbstractVector}
    T_c::T
    rho_c::T
    R_univ::T
    M::T
    T_ref::T
    a_coeffs::V
    k_coeffs::V
    N::V
    t::V
    d::V
    p::V
    α::V
    β::V
    γ::V
    D::V
    p_c::T
    vapour_N::V
    vapour_k::V
    T_triple::T
    p_triple::T
    liquid_multiplier::T
end
Adapt.@adapt_structure HelmholtzFluidConstants



### Lambdas - functions that are useful for quick calculations of various properties

function lambda_0_01(δ::F, τ::F, constants, fluid) where F <: AbstractFloat
    return δ * d_alpha_0_d_delta(δ, τ, constants, fluid)
end
function lambda_0_02(δ::F, τ::F, constants, fluid) where F <: AbstractFloat
    return (δ^2) * d2_alpha_0_d_delta2(δ, τ, constants, fluid)
end

function lambda_0_10(δ::F, τ::F, constants, fluid) where F <: AbstractFloat
    return τ * d_alpha_0_d_tau(δ, τ, constants, fluid)
end
function lambda_0_20(δ::F, τ::F, constants, fluid) where F <: AbstractFloat
    return (τ^2) * d2_alpha_0_d_tau2(δ, τ, constants, fluid)
end

function lambda_0_11(δ::F, τ::F, constants, fluid) where F <: AbstractFloat
    return zero(F) # delta * tau * d2alpha_0 / ddelta*dtau
end




function lambda_r_01(δ::F, τ::F, constants, fluid) where F <: AbstractFloat
    return δ * d_alpha_r_d_delta(δ, τ, constants, fluid)
end
function lambda_r_02(δ::F, τ::F, constants, fluid) where F <: AbstractFloat
    return (δ^2) * d2_alpha_r_d_delta2(δ, τ, constants, fluid)
end

function lambda_r_10(δ::F, τ::F, constants, fluid) where F <: AbstractFloat
    return τ * d_alpha_r_d_tau(δ, τ, constants, fluid)
end
function lambda_r_20(δ::F, τ::F, constants, fluid) where F <: AbstractFloat
    return (τ^2) * d2_alpha_r_d_tau2(δ, τ, constants, fluid)
end

function lambda_r_11(δ::F, τ::F, constants, fluid) where F <: AbstractFloat
    return δ * τ * d2_alpha_r_d_delta_d_tau(δ, τ, constants, fluid)
end




function lambda_total_01(δ::F, τ::F, constants, fluid) where F <: AbstractFloat
    return lambda_0_01(δ, τ, constants, fluid) + lambda_r_01(δ, τ, constants, fluid)
end
function lambda_total_02(δ::F, τ::F, constants, fluid) where F <: AbstractFloat
    return lambda_0_02(δ, τ, constants, fluid) + lambda_r_02(δ, τ, constants, fluid)
end

function lambda_total_10(δ::F, τ::F, constants, fluid) where F <: AbstractFloat
    return lambda_0_10(δ, τ, constants, fluid) + lambda_r_10(δ, τ, constants, fluid)
end
function lambda_total_20(δ::F, τ::F, constants, fluid) where F <: AbstractFloat
    return lambda_0_20(δ, τ, constants, fluid) + lambda_r_20(δ, τ, constants, fluid)
end

function lambda_total_11(δ::F, τ::F, constants, fluid) where F <: AbstractFloat
    return lambda_0_11(δ, τ, constants, fluid) + lambda_r_11(δ, τ, constants, fluid)
end


### CORE CALCULATIONS SECTION

"""Computes pressure (in Pa) for a given temperature (K) and molar density (mol/m^3)."""
function compute_pressure(T::F, rho::F, constants, fluid) where F <: AbstractFloat
    (; T_c, rho_c, R_univ) = constants
    τ = T_c / T
    δ = rho / rho_c
    
    Z = F(1) + lambda_r_01(δ, τ, constants, fluid)
    return Z * rho * R_univ * T
end

"""Computes the isochoric (constant volume) heat capacity in J/(mol*K)."""
function c_v(δ::F, τ::F, constants, fluid) where F <: AbstractFloat
    (; R_univ) = constants
    return -R_univ * lambda_total_20(δ, τ, constants, fluid)
end

"""Computes the isobaric (constant pressure) heat capacity in J/(mol*K)."""
function c_p(δ::F, τ::F, constants, fluid) where F <: AbstractFloat
    (; R_univ) = constants
    cv_term = c_v(δ, τ, constants, fluid)
    numerator = (one(F) + lambda_total_01(δ, τ, constants, fluid) - lambda_total_11(δ, τ, constants, fluid))^2
    denominator = one(F) + F(2) * lambda_total_01(δ, τ, constants, fluid) + lambda_total_02(δ, τ, constants, fluid)
    return cv_term + (R_univ * (numerator / denominator))
end

"""Computes the isothermal compressibility in 1/Pa (required for thermal conductivity computation)."""
function k_T(T::F, rho::F, constants, fluid) where F <: AbstractFloat
    (; T_c, rho_c, R_univ) = constants
    τ = T_c / T
    δ = rho / rho_c
    term1 = rho * R_univ * T
    term2 = one(F) + F(2) * lambda_r_01(δ, τ, constants, fluid) + lambda_r_02(δ, τ, constants, fluid)
    return one(F) / (term1 * term2)
end

"""Computes the ideal-gas isochoric heat capacity (Cv0) in J/(mol*K) at a given temperature."""
function cv0_calc(T::F, constants, fluid) where F <: AbstractFloat
    (; T_c, R_univ) = constants
    τ = T_c / T
    return -R_univ * lambda_0_20(one(F), τ, constants, fluid)
end

"""Computes the ideal-gas internal energy (u0) in J/mol at a given temperature."""
function u0_calc(T::F, constants, fluid) where F <: AbstractFloat
    (; T_c, R_univ) = constants
    τ = T_c / T
    return R_univ * T * lambda_0_10(one(F), τ, constants, fluid)
end

"""Computes the ideal-gas enthalpy (h0) in J/mol at a given temperature."""
function h0_calc(T::F, constants, fluid) where F <: AbstractFloat
    (; R_univ) = constants
    return u0_calc(T, constants, fluid) + R_univ * T
end

"""Computes the total internal energy (ideal + residual) in J/mol."""
function internal_energy_calc(T::F, δ::F, τ::F, constants, fluid) where F <: AbstractFloat
    (; R_univ) = constants
    u_ideal = u0_calc(T, constants, fluid)
    u_residual = R_univ * T * lambda_r_10(δ, τ, constants, fluid)
    return u_ideal + u_residual
end

"""Computes the total enthalpy (ideal + residual) in J/mol."""
function enthalpy_calc(T::F, δ::F, τ::F, constants, fluid) where F <: AbstractFloat
    (; R_univ) = constants
    h_ideal = h0_calc(T, constants, fluid)
    h_residual = R_univ * T * (lambda_r_10(δ, τ, constants, fluid) + lambda_r_01(δ, τ, constants, fluid))
    return h_ideal + h_residual
end

"""Computes the total entropy (ideal + residual) in J/(mol*K)."""
function entropy_calc(δ::F, τ::F, constants, fluid) where F <: AbstractFloat
    (; R_univ) = constants
    term1 = lambda_total_10(δ, τ, constants, fluid)
    term2 = alpha_0(δ, τ, constants, fluid) + alpha_r(δ, τ, constants, fluid)
    return R_univ * (term1 - term2)
end

"""Computes the isobaric expansion coefficient in 1/K."""
function beta_calc(T::F, δ::F, τ::F, constants, fluid) where F <: AbstractFloat
    numerator = one(F) + lambda_r_01(δ, τ, constants, fluid) - lambda_r_11(δ, τ, constants, fluid)
    denominator = one(F) + F(2)*lambda_r_01(δ, τ, constants, fluid) + lambda_r_02(δ, τ, constants, fluid)
    
    return (one(F) / T) * (numerator / denominator)
end




"""Computes the Gibbs free energy in J/mol for a given state."""
function gibbs_free_energy(T::F, rho::F, constants, fluid) where F <: AbstractFloat
    (; T_c, rho_c, R_univ) = constants

    τ = T_c / T
    δ = rho / rho_c

    alpha_val = alpha_0(δ, τ, constants, fluid) + alpha_r(δ, τ, constants, fluid)
    
    p_val = compute_pressure(T, rho, constants, fluid)
    Z = p_val / (rho * R_univ * T)

    # Calculate dimensional Gibbs free energy
    g = (alpha_val + Z) * R_univ * T
    
    return g
end


"""
Iteratively solves for molar density (mol/m3) given a temperature (K) and pressure (Pa).
Uses the Newton-Raphson method with an initial guess `rho_guess`.
"""
# Advanced function that accounts for discontinuities
function find_density_advanced(T::F, P_target::F, rho_guess::F, constants, fluid;
    max_iter=100, tol=1.0e-8) where F <: AbstractFloat

    (; T_c, p_c, rho_c, R_univ) = constants
    TOL = 1e-4

    if (abs(T - T_c) < TOL) && (abs(P_target - p_c) < TOL)
        return rho_c # Exception case, return critical density straight away (at T=T_crit and p=p_crit)
    end

    τ = T_c / T
    RT = R_univ * T

    # Initial guess for density
    rho = rho_guess

    for it in 1:max_iter
        δ = rho / rho_c
        
        p_calc = compute_pressure(T, rho, constants, fluid)
        f = p_calc - P_target

        if abs(f / P_target) < tol
            return rho
        end

        # Analytical derivative dp/drho
        dp_drho = RT * (one(F) + F(2) * lambda_r_01(δ, τ, constants, fluid) + lambda_r_02(δ, τ, constants, fluid))
        
        # Newton-Raphson step
        step = f / dp_drho
        relaxation_coeff = one(F)
        rho_new = rho - ( relaxation_coeff * step )
        
        # Check for nonphysical results
        if rho_new <= 0 || !isfinite(rho_new)

            error("[Density Solver] Nonphysical density value was obtained")
        end
        rho = rho_new
    end

    @warn "[Density Solver] Did not converge in $max_iter iterations! T: $T, p: $P_target"
end


"""Calculates the saturation pressure (in Pa) for a given temperature using a simplified ancillary equation."""
function vapour_pressure_ancillary(T::F, constants, fluid) where F <: AbstractFloat
    (; T_c, p_c, vapour_N, vapour_k) = constants

    # Coefficients and exponents from Table 8
    N = vapour_N
    k = vapour_k
    
    θ = one(F) - T / T_c
    
    sum_val = zero(F)
    for i in eachindex(N)
        sum_val += N[i] * (θ^k[i])
    end
    
    ln_pr = (T_c / T) * sum_val
    
    return exp(ln_pr) * p_c
end

"""Calculates the derivative of saturation pressure with respect to temperature (dPsat/dT) using the ancillary equation."""
function dPsat_dT(T::F, p_pair::F, constants, fluid) where F <: AbstractFloat
    (; T_c, vapour_N, vapour_k) = constants
    
    # Coefficients and exponents from Table 8
    N = vapour_N
    k = vapour_k
    
    θ = one(F) - T / T_c
    
    S_T = zero(F)
    sum_dS_dT_term = zero(F)
    for i in eachindex(N)
        S_T += N[i] * (θ^k[i])
        sum_dS_dT_term += N[i] * k[i] * (θ^(k[i] - one(F)))
    end

    derivative_term = ( (-T_c / (T^2.0)) * S_T ) - ( (one(F) / T) * sum_dS_dT_term )
    
    return p_pair * derivative_term # apply chain rule
end


"""Iteratively solves for the saturation temperature (in K) for a given pressure (Pa) using the ancillary equation."""
function find_saturation_temperature(P_target::F, constants, fluid; max_iter=100, tol=1.0e-7) where F <: AbstractFloat
    (; T_c, p_c, T_triple, p_triple, vapour_N) = constants
    # Newton-Raphson method, similar to density

    if P_target > p_c # Check if there is such thing as saturation temperature
        println("Pressure ($P_target Pa) is above the critical pressure ($p_c Pa). Saturation temperature is not defined!!!")
        return zero(F)
    end
    if P_target < p_triple
         println("Pressure ($P_target Pa) is below the triple point pressure ($p_triple Pa). Saturation temperature is not defined11!")
        return zero(F)
    end

    # Initial guess for temp using a simplified inversion of the ancillary equation
    N1 = vapour_N[1] # First coefficient from the ancillary equation
    T_guess = T_c / (one(F) + log(P_target / p_c) / N1)
    
    T = clamp(T_guess, T_triple, T_c - 1e-6) # Make sure guess is physical
   
    for it in 1:max_iter
        # Calculate residual
        P_calc = vapour_pressure_ancillary(T, constants, fluid)
        f = P_calc - P_target

        # Check for convergence
        if abs(f / P_target) < tol
            return T
        end

        # Calculate derivative for Newton-Raphson step
        dP_dT = dPsat_dT(T, P_calc, constants, fluid)
        
        if abs(dP_dT) < 1e-9 # Avoid division by zero
             error("T_SAT : DERIVATIVE IS ALMOST ZERO")
        end

        # Newton-Raphson step
        T_new = T - f / dP_dT
        
        T = clamp(T_new, T_triple, T_c - 1e-6)
    end
    
    if abs(T - T_c) < F(1e-4)
        return T_c # exception case where p=p_crit and T=t_crit
    else
        error("[T_saturation Solver] Did not converge in $max_iter iterations. TEMP: $T")
    end
end


"""
Calculates saturation properties (pressure, liquid density, vapor density) at a given temperature.
Uses a secant solver to find the pressure where the liquid and vapor Gibbs free energies are equal.
"""
function find_saturation_properties(T::F, pressure::F, constants, fluid; max_iter=30, tol=1.0e-7) where F <: AbstractFloat
    (; T_c, rho_c, R_univ, liquid_multiplier) = constants

    if T >= T_c # Ensure the temperature is in the valid range (below critical)
        error("Temperature must be below the critical temperature for saturation calculation.")
    end

    p_guess = vapour_pressure_ancillary(T, constants, fluid) # This function would vary depending on fluid

    # Secant method for iterative solver
    p_sat = p_guess
    p_prev = p_guess * 0.999 # Slightly perturb the previous pressure value for first step

    g_diff_current = zero(F)
    g_diff_prev = zero(F)

    for it in 1:max_iter
        p_ideal_gas = p_sat / (R_univ * T)
        p_multiplied = liquid_multiplier * rho_c

        rho_l = find_density_advanced(T, p_sat, p_multiplied, constants, fluid) # Higher guess for liquid
        rho_v = find_density_advanced(T, p_sat, p_ideal_gas, constants, fluid) # Ideal gas guess for vapour

        # Check if roots were found
        if isnan(rho_l) || isnan(rho_v)
            error("Failed to find liquid or vapor density root at T=$T K, P=$p_sat Pa...")
        end

        # Compute difference in gibbs free energy
        g_l = gibbs_free_energy(T, rho_l, constants, fluid)
        g_v = gibbs_free_energy(T, rho_v, constants, fluid)
        g_diff_current = g_l - g_v

        # Convergence check based on gibbs energy
        if abs(g_diff_current / g_l) < tol
            # println("Saturation solver converged in $it iterations.")

            T_sat = zero(F)
            T_sat = find_saturation_temperature(pressure, constants, fluid)

            return (p_sat, T_sat, rho_l, rho_v)
        end

        # Secant method step
        if it > 1
            if abs(g_diff_current - g_diff_prev) < 1e-9 # Avoid division by zero
                p_next = p_sat * 1.001
            else
                p_next = p_sat - g_diff_current * (p_sat - p_prev) / (g_diff_current - g_diff_prev)
            end
            
            # Update values for next itration
            p_prev = p_sat
            g_diff_prev = g_diff_current
            p_sat = p_next
        else
            
            p_prev = p_sat
            g_diff_prev = g_diff_current
            
            p_sat = p_guess * 1.001
        end

        if p_sat <= 0 # Check for -ve temps
            error("Solver stepped to a -ve pressure.")
        end
    end

    error("[Saturation Properties Solver] Did not converge in $max_iter iterations.")
end



"""
Computes a set of key thermodynamic properties from molar density and temperature,
and converts them from molar to mass-specific units.
"""
function params_computation(rho_mol::F, T::F, constants, fluid) where F <: AbstractFloat
    (; T_c, rho_c, M, T_ref) = constants

    τ = T_c / T
    δ = rho_mol / rho_c

    cv_mol = c_v(δ, τ, constants, fluid)
    cp_mol = c_p(δ, τ, constants, fluid)
    kT = k_T(T, rho_mol, constants, fluid)
    kT_ref = k_T(T_ref, rho_mol, constants, fluid)

    internal_energy_mol = internal_energy_calc(T, δ, τ, constants, fluid)
    enthalpy_mol = enthalpy_calc(T, δ, τ, constants, fluid)
    entropy_mol = entropy_calc(δ, τ, constants, fluid)
    
    beta_mol = beta_calc(T, δ, τ, constants, fluid)

    conversion_factor = one(F) / (M * F(1e3))

    rho = rho_mol * M

    cv = cv_mol*conversion_factor
    cp = cp_mol*conversion_factor
    internal_energy = internal_energy_mol*conversion_factor
    enthalpy = enthalpy_mol*conversion_factor
    entropy = entropy_mol*conversion_factor
    beta = beta_mol*conversion_factor # NOT TESTED!!!!!

    return rho, cv, cp, kT, kT_ref, internal_energy, enthalpy, entropy, beta
end



"""
Main high-level wrapper. Determines the fluid phase (liquid, vapor, two-phase, or supercritical)
and computes a comprehensive set of thermodynamic properties for the given state (T, P).
"""
function EOS_wrapper(fluid::HelmholtzEnergyFluid, T::F, pressure::F, constants) where F <: AbstractFloat
    
    (; T_c, rho_c, R_univ, M, T_ref, p_c, liquid_multiplier) = constants

    TOL = 1e-4
    # alpha_tol = 1e-8

    rho_mol = zero(F)
    rho_list = [zero(F), zero(F)]

    latentHeat = zero(F)
    T_sat = zero(F)

    # m_lv = zero(F)
    # m_vl = zero(F)

    rho_vals = [zero(F), zero(F)]
    cv_vals = [zero(F), zero(F)]
    cp_vals = [zero(F), zero(F)]
    kT_vals = [zero(F), zero(F)]
    kT_ref_vals = [zero(F), zero(F)]
    internal_energy_vals = [zero(F), zero(F)]
    enthalpy_vals = [zero(F), zero(F)]
    entropy_vals = [zero(F), zero(F)]
    beta_vals = [zero(F), zero(F)]

    if T >= T_c # Account for supercritical fluid / superheated vapour state
        rho_guess = pressure / (R_univ * T) # Ideal gas guess
        rho_mol = find_density_advanced(T, pressure, rho_guess, constants, fluid)
        rho_list = [rho_mol, rho_mol] # if T > T_crit, we want to return two identical densities

    else # Else it is liquid/vapour
        (P_sat, T_sat, rho_l_sat, rho_v_sat) = find_saturation_properties(T, pressure, constants, fluid)

        if ( abs(pressure - P_sat) / P_sat ) < TOL # TWO PHASE REGION, pressure matched saturation line
            rho_mol_liquid = find_density_advanced(T, pressure, rho_l_sat, constants, fluid) #maybe T_sat is better?
            rho_mol_vapour = find_density_advanced(T, pressure, rho_v_sat, constants, fluid) #maybe T_sat is better?

            rho_list[1] = rho_mol_liquid 
            rho_list[2] = rho_mol_vapour

        elseif pressure < P_sat # VAPOUR REGION
            rho_guess = pressure / (R_univ * T) # Ideal gas guess
            rho_guess_ = liquid_multiplier * rho_c # Higher density guess
            rho_mol = find_density_advanced(T, pressure, rho_guess, constants, fluid)
            rho_liquid = find_density_advanced(T_sat, pressure, rho_guess_, constants, fluid)

            rho_list[1] = rho_liquid
            rho_list[2] = rho_mol

            # m_lv = c_τ_evap * alpha * rho_liquid * ( (T - T_sat)/T_sat ) # ASSUME alpha=1 is liquid

        elseif pressure > P_sat # LIQUID REGION
            rho_guess = liquid_multiplier * rho_c # Higher density guess
            rho_guess_ = pressure / (R_univ * T) # Ideal gas guess
            rho_mol = find_density_advanced(T, pressure, rho_guess, constants, fluid)
            rho_vapour = find_density_advanced(T_sat, pressure, rho_guess_, constants, fluid)

            rho_list[1] = rho_mol
            rho_list[2] = rho_vapour

            # m_vl = c_τ_cond * (one(F)-alpha) * rho_vapour * ( (T_sat - T)/T_sat )# ASSUME alpha=0 is liquid vapour
        end

        if isnan(rho_mol)
            error("Failed to find a valid density for the given state.")
        end
    end

    for i in eachindex(rho_list)
        rho_vals[i], cp_vals[i], cv_vals[i], kT_vals[i], kT_ref_vals[i], 
        internal_energy_vals[i], enthalpy_vals[i], entropy_vals[i], beta_vals[i] = params_computation(rho_list[i], T, constants, fluid)
    end

    latentHeat = enthalpy_vals[2] - enthalpy_vals[1] #enthalpy_V - enthalpy_L

    return rho_vals, cp_vals, cv_vals, kT_vals, 
            kT_ref_vals, internal_energy_vals, enthalpy_vals, beta_vals, entropy_vals, latentHeat, T_sat#, m_lv, m_vl
end






##### h inversion into T in case needed in the future for energy equation
#####
# function h_into_T(h_target::F, P_target::F, T_guess::F, constants, fluid;
#                     max_iter=20, tol=1.0e-7) where F <: AbstractFloat

#     (; T_c, rho_c, R_univ, M_H2, T_ref) = constants
#     # h_target is h_n+1
#     # p_target is p_n+1
#     # T_guess is T_n

#     T = T_guess # Start with an initial guess
    
#     for it in 1:max_iter

#         rho_current = find_density(T, P_target, constants, fluid) # A MORE ADVANCED APPROACH IS REQUIRED!

#         if isnan(rho_current)
#             error("Density solver failed during T inversion.")
#         end

#         τ = T_c / T
#         δ = rho_current / rho_c #mol value!

#         h_calc = enthalpy_calc(T, δ, τ, constants, fluid) #WARNING : need to check units!!!

#         # Calculate the residual
#         f = h_calc - h_target
#         if abs(f / h_target) < tol
#             return T # Converged
#         end

#         # Calculate the derivative (cv)
#         cp_val = c_p(T, rho_current, constants, fluid)
#         if abs(cp_val) < 1e-9
#             error("Derivative (cv) is near zero.")
#         end

#         # Newton's step
#         T = T - f / cp_val
#     end

#     error("Temperature inversion failed to converge.")
# end




##### u inversion into T in case needed in the future for energy equation
#####
# function u_into_T(u_target::F, P_target::F, T_guess::F, constants, fluid::constants_EoS_N2;
#                     max_iter=20, tol=1.0e-7)

#     (; T_c, rho_c) = constants
#     # u_target is u_n+1
#     # p_target is p_n+1
#     # T_guess is T_n

#     T = T_guess # Start with an initial guess
    
#     for it in 1:max_iter

#         rho_current = find_density(T, P_target, constants, fluid) # A MORE ADVANCED APPROACH IS REQUIRED!

#         if isnan(rho_current)
#             error("Density solver failed during T inversion.")
#         end

#         τ = T_c / T
#         δ = rho_current / rho_c #mol value!

#         u_calc = internal_energy_calc(T, δ, τ, constants, fluid) #WARNING : need to check units!!!

#         # Calculate the residual
#         f = u_calc - u_target
#         if abs(f / u_target) < tol
#             return T # Converged
#         end

#         # Calculate the derivative (cv)
#         cv_val = c_v(T, rho_current, constants, fluid)
#         if abs(cv_val) < 1e-9
#             error("Derivative (cv) is near zero.")
#         end

#         # Newton's step
#         T = T - f / cv_val
#     end

#     error("Temperature inversion failed to converge.")
# end