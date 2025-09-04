export lambda_0_01,
       lambda_0_02,
       lambda_0_10,
       lambda_0_20,
       lambda_0_11,
       lambda_r_01,
       lambda_r_02,
       lambda_r_10,
       lambda_r_20,
       lambda_r_11,
       lambda_total_01,
       lambda_total_02,
       lambda_total_10,
       lambda_total_20,
       lambda_total_11,
       compute_pressure,
       c_v,
       c_p,
       k_T,
       cv0_calc,
       u0_calc,
       h0_calc,
       internal_energy_calc,
       enthalpy_calc,
       entropy_calc,
       beta_calc,
       gibbs_free_energy,
       find_density_advanced,
       params_computation,
       vapour_pressure_ancillary,
       dPsat_dT,
       find_saturation_properties,
       find_saturation_temperature




### Lambdas

function lambda_0_01(δ::F, τ::F, constants) where F <: AbstractFloat
    return δ * d_alpha_0_d_delta(δ, τ, constants)
end
function lambda_0_02(δ::F, τ::F, constants) where F <: AbstractFloat
    return (δ^2) * d2_alpha_0_d_delta2(δ, τ, constants)
end

function lambda_0_10(δ::F, τ::F, constants) where F <: AbstractFloat
    return τ * d_alpha_0_d_tau(δ, τ, constants)
end
function lambda_0_20(δ::F, τ::F, constants) where F <: AbstractFloat
    return (τ^2) * d2_alpha_0_d_tau2(δ, τ, constants)
end

function lambda_0_11(δ::F, τ::F, constants) where F <: AbstractFloat
    return zero(F) # delta * tau * d2alpha_0 / ddelta*dtau
end




function lambda_r_01(δ::F, τ::F, constants) where F <: AbstractFloat
    return δ * d_alpha_r_d_delta(δ, τ, constants)
end
function lambda_r_02(δ::F, τ::F, constants) where F <: AbstractFloat
    return (δ^2) * d2_alpha_r_d_delta2(δ, τ, constants)
end

function lambda_r_10(δ::F, τ::F, constants) where F <: AbstractFloat
    return τ * d_alpha_r_d_tau(δ, τ, constants)
end
function lambda_r_20(δ::F, τ::F, constants) where F <: AbstractFloat
    return (τ^2) * d2_alpha_r_d_tau2(δ, τ, constants)
end

function lambda_r_11(δ::F, τ::F, constants) where F <: AbstractFloat
    return δ * τ * d2_alpha_r_d_delta_d_tau(δ, τ, constants)
end




function lambda_total_01(δ::F, τ::F, constants) where F <: AbstractFloat
    return lambda_0_01(δ, τ, constants) + lambda_r_01(δ, τ, constants)
end
function lambda_total_02(δ::F, τ::F, constants) where F <: AbstractFloat
    return lambda_0_02(δ, τ, constants) + lambda_r_02(δ, τ, constants)
end

function lambda_total_10(δ::F, τ::F, constants) where F <: AbstractFloat
    return lambda_0_10(δ, τ, constants) + lambda_r_10(δ, τ, constants)
end
function lambda_total_20(δ::F, τ::F, constants) where F <: AbstractFloat
    return lambda_0_20(δ, τ, constants) + lambda_r_20(δ, τ, constants)
end

function lambda_total_11(δ::F, τ::F, constants) where F <: AbstractFloat
    return lambda_0_11(δ, τ, constants) + lambda_r_11(δ, τ, constants)
end


### CORE CALCULATIONS SECTION


function compute_pressure(T::F, rho::F, constants) where F <: AbstractFloat
    (; T_c, rho_c, R_univ) = constants
    τ = T_c / T
    δ = rho / rho_c
    
    Z = F(1) + lambda_r_01(δ, τ, constants)
    return Z * rho * R_univ * T
end

function c_v(δ::F, τ::F, constants) where F <: AbstractFloat
    (; R_univ) = constants
    return -R_univ * lambda_total_20(δ, τ, constants)
end

function c_p(δ::F, τ::F, constants) where F <: AbstractFloat
    (; R_univ) = constants
    cv_term = c_v(δ, τ, constants)
    numerator = (one(F) + lambda_total_01(δ, τ, constants) - lambda_total_11(δ, τ, constants))^2
    denominator = one(F) + F(2) * lambda_total_01(δ, τ, constants) + lambda_total_02(δ, τ, constants)
    return cv_term + (R_univ * (numerator / denominator))
end

function k_T(T::F, rho::F, constants) where F <: AbstractFloat
    (; T_c, rho_c, R_univ) = constants
    τ = T_c / T
    δ = rho / rho_c
    term1 = rho * R_univ * T
    term2 = one(F) + F(2) * lambda_r_01(δ, τ, constants) + lambda_r_02(δ, τ, constants)
    return one(F) / (term1 * term2)
end

function cv0_calc(T::F, constants) where F <: AbstractFloat
    (; T_c, R_univ) = constants
    τ = T_c / T
    return -R_univ * lambda_0_20(one(F), τ, constants)
end

function u0_calc(T::F, constants) where F <: AbstractFloat
    (; T_c, R_univ) = constants
    τ = T_c / T
    return R_univ * T * lambda_0_10(one(F), τ, constants)
end

function h0_calc(T::F, constants) where F <: AbstractFloat
    (; R_univ) = constants
    return u0_calc(T, constants) + R_univ * T
end

function internal_energy_calc(T::F, δ::F, τ::F, constants) where F <: AbstractFloat
    (; R_univ) = constants
    u_ideal = u0_calc(T, constants)
    u_residual = R_univ * T * lambda_r_10(δ, τ, constants)
    return u_ideal + u_residual
end

function enthalpy_calc(T::F, δ::F, τ::F, constants) where F <: AbstractFloat
    (; R_univ) = constants
    h_ideal = h0_calc(T, constants)
    h_residual = R_univ * T * (lambda_r_10(δ, τ, constants) + lambda_r_01(δ, τ, constants))
    return h_ideal + h_residual
end

function entropy_calc(δ::F, τ::F, constants) where F <: AbstractFloat
    (; R_univ) = constants
    term1 = lambda_total_10(δ, τ, constants)
    term2 = alpha_0(δ, τ, constants) + alpha_r(δ, τ, constants)
    return R_univ * (term1 - term2)
end

function beta_calc(T::F, δ::F, τ::F, constants) where F <: AbstractFloat
    numerator = one(F) + lambda_r_01(δ, τ, constants) - lambda_r_11(δ, τ, constants)
    denominator = one(F) + F(2)*lambda_r_01(δ, τ, constants) + lambda_r_02(δ, τ, constants)
    
    return (one(F) / T) * (numerator / denominator)
end





function gibbs_free_energy(T::F, rho::F, constants) where F <: AbstractFloat
    (; T_c, rho_c, R_univ) = constants

    τ = T_c / T
    δ = rho / rho_c

    alpha_val = alpha_0(δ, τ, constants) + alpha_r(δ, τ, constants)
    
    p_val = compute_pressure(T, rho, constants)
    Z = p_val / (rho * R_univ * T)

    # Calculate dimensional Gibbs free energy
    g = (alpha_val + Z) * R_univ * T
    
    return g
end



# Advanced function that accounts for discontinuities
function find_density_advanced(T::F, P_target::F, rho_guess::F, constants;
    max_iter=100, tol=1.0e-8) where F <: AbstractFloat

    (; T_c, rho_c, R_univ) = constants

    τ = T_c / T
    RT = R_univ * T

    # Initial guess for density
    rho = rho_guess

    for it in 1:max_iter
        δ = rho / rho_c
        
        p_calc = compute_pressure(T, rho, constants)
        f = p_calc - P_target

        if abs(f / P_target) < tol
            return rho
        end

        # Analytical derivative dp/drho
        dp_drho = RT * (one(F) + F(2) * lambda_r_01(δ, τ, constants) + lambda_r_02(δ, τ, constants))
        
        # Newton-Raphson step
        step = f / dp_drho
        relaxation_coeff = one(F)
        rho_new = rho - ( relaxation_coeff * step )
        
        # Check for nonphysical results
        if rho_new <= 0 || !isfinite(rho_new)
            return NaN 
        end
        rho = rho_new
    end

    error("[Density Solver] Did not converge in $max_iter iterations! T: $T, p: $P_target")
end



function vapour_pressure_ancillary(T::F, constants) where F <: AbstractFloat
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


function dPsat_dT(T::F, p_pair::F, constants) where F <: AbstractFloat
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


function find_saturation_temperature(P_target::F, constants; max_iter=100, tol=1.0e-7) where F <: AbstractFloat
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
        P_calc = vapour_pressure_ancillary(T, constants)
        f = P_calc - P_target

        # Check for convergence
        if abs(f / P_target) < tol
            return T
        end

        # Calculate derivative for Newton-Raphson step
        dP_dT = dPsat_dT(T, P_calc, constants)
        
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



function find_saturation_properties(T::F, pressure::F, constants; max_iter=30, tol=1.0e-7) where F <: AbstractFloat
    (; T_c, rho_c, R_univ, liquid_multiplier) = constants

    if T >= T_c # Ensure the temperature is in the valid range (below critical)
        error("Temperature must be below the critical temperature for saturation calculation.")
    end

    p_guess = vapour_pressure_ancillary(T, constants) # This function would vary depending on fluid

    # Secant method for iterative solver
    p_sat = p_guess
    p_prev = p_guess * 0.999 # Slightly perturb the previous pressure value for first step

    g_diff_current = zero(F)
    g_diff_prev = zero(F)

    for it in 1:max_iter
        p_ideal_gas = p_sat / (R_univ * T)
        p_multiplied = liquid_multiplier * rho_c

        rho_l = find_density_advanced(T, p_sat, p_multiplied, constants) # Higher guess for liquid
        rho_v = find_density_advanced(T, p_sat, p_ideal_gas, constants) # Ideal gas guess for vapour

        # Check if roots were found
        if isnan(rho_l) || isnan(rho_v)
            error("Failed to find liquid or vapor density root at T=$T K, P=$p_sat Pa...")
        end

        # Compute difference in gibbs free energy
        g_l = gibbs_free_energy(T, rho_l, constants)
        g_v = gibbs_free_energy(T, rho_v, constants)
        g_diff_current = g_l - g_v

        # Convergence check based on gibbs energy
        if abs(g_diff_current / g_l) < tol
            # println("Saturation solver converged in $it iterations.")

            T_sat = zero(F)
            T_sat = find_saturation_temperature(pressure, constants)

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




function params_computation(rho_mol::F, T::F, constants) where F <: AbstractFloat
    (; T_c, rho_c, M, T_ref) = constants

    τ = T_c / T
    δ = rho_mol / rho_c

    cv_mol = c_v(δ, τ, constants)
    cp_mol = c_p(δ, τ, constants)
    kT = k_T(T, rho_mol, constants)
    kT_ref = k_T(T_ref, rho_mol, constants)

    internal_energy_mol = internal_energy_calc(T, δ, τ, constants)
    enthalpy_mol = enthalpy_calc(T, δ, τ, constants)
    entropy_mol = entropy_calc(δ, τ, constants)
    
    beta_mol = beta_calc(T, δ, τ, constants)

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










##### h inversion into T in case needed in the future for energy equation
#####
# function h_into_T(h_target::F, P_target::F, T_guess::F, constants;
#                     max_iter=20, tol=1.0e-7) where F <: AbstractFloat

#     (; T_c, rho_c, R_univ, M_H2, T_ref) = constants
#     # h_target is h_n+1
#     # p_target is p_n+1
#     # T_guess is T_n

#     T = T_guess # Start with an initial guess
    
#     for it in 1:max_iter

#         rho_current = find_density(T, P_target, constants) # A MORE ADVANCED APPROACH IS REQUIRED!

#         if isnan(rho_current)
#             error("Density solver failed during T inversion.")
#         end

#         τ = T_c / T
#         δ = rho_current / rho_c #mol value!

#         h_calc = enthalpy_calc(T, δ, τ, constants) #WARNING : need to check units!!!

#         # Calculate the residual
#         f = h_calc - h_target
#         if abs(f / h_target) < tol
#             return T # Converged
#         end

#         # Calculate the derivative (cv)
#         cp_val = c_p(T, rho_current, constants)
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
# function u_into_T(u_target::F, P_target::F, T_guess::F, constants::constants_EoS_N2;
#                     max_iter=20, tol=1.0e-7)

#     (; T_c, rho_c) = constants
#     # u_target is u_n+1
#     # p_target is p_n+1
#     # T_guess is T_n

#     T = T_guess # Start with an initial guess
    
#     for it in 1:max_iter

#         rho_current = find_density(T, P_target, constants) # A MORE ADVANCED APPROACH IS REQUIRED!

#         if isnan(rho_current)
#             error("Density solver failed during T inversion.")
#         end

#         τ = T_c / T
#         δ = rho_current / rho_c #mol value!

#         u_calc = internal_energy_calc(T, δ, τ, constants) #WARNING : need to check units!!!

#         # Calculate the residual
#         f = u_calc - u_target
#         if abs(f / u_target) < tol
#             return T # Converged
#         end

#         # Calculate the derivative (cv)
#         cv_val = c_v(T, rho_current, constants)
#         if abs(cv_val) < 1e-9
#             error("Derivative (cv) is near zero.")
#         end

#         # Newton's step
#         T = T - f / cv_val
#     end

#     error("Temperature inversion failed to converge.")
# end