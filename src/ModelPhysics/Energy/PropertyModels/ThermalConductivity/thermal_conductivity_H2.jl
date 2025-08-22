export thermal_conductivity_H2

###Refer to "Correlation of the Thermal Conductivity of Normal and Parahydrogen
###                              from the Triple Point to 1000 K and up to 100MPa", 2011

using Printf
using XCALibre

struct constants_k_H2
    T_c::Float64
    rho_c::Float64
    P_CRIT::Float64
    A1::Vector{Float64}
    A2::Vector{Float64}
    B1::Vector{Float64}
    B2::Vector{Float64}
    C1::Float64
    C2::Float64
    C3::Float64
    R_D::Float64
    ν::Float64
    γ_crit::Float64
    xi_0::Float64
    GAMMA_0::Float64
    qD::Float64
    T_ref::Float64
    k_B::Float64
end

function lambda0(T::Float64, constants::constants_k_H2)
    (; T_c, A1, A2) = constants

    T_r = T / T_c
    numerator = sum(A1[i+1] * (T_r)^i for i in 0:7)
    denominator = sum(A2[i+1] *(T_r)^i for i in 0:6)
    return numerator / denominator
end


function delta_lambda(rho::Float64, T::Float64, constants::constants_k_H2)
    (; T_c, rho_c, B1, B2) = constants

    T_r = T / T_c
    rho_r = rho / rho_c

    term_sum = 0.0
    for i in 1:5
        term_sum += (B1[i] + (B2[i] * T_r) ) * (rho_r)^i
    end
    return term_sum
end


function delta_lambda_c_empirical(rho::Float64, T::Float64, constants::constants_k_H2) #The easy version
    (; T_c, rho_c, C1, C2, C3) = constants

    delta_T_c   = (T / T_c) - 1
    delta_rho_c   = (rho / rho_c) - 1
    denominator  = C2 + abs(delta_T_c)

    if denominator <= 0.0
        return 0.0
    else
        return (C1 / denominator) * exp(-(C3 * delta_rho_c)^2)
    end
end

function xi(rho::Float64, T::Float64, kT::Float64, kT_ref::Float64, constants::constants_k_H2)
    (; T_c, rho_c, P_CRIT, A1, A2, B1, B2, C1, C2, C3, R_D, ν, γ_crit, xi_0, GAMMA_0, qD, T_ref, k_B) = constants
    # kT = (1/rho) * (d rho / d p) at constant T
    # kT is evaluated at T passed into delta_lambda_c, while kT_ref evaluated at T_ref
    
    nu_div_gamma = ν/γ_crit

    term1 = xi_0 * ((P_CRIT*rho)/(GAMMA_0*(rho_c^2)))^nu_div_gamma

    bracket_term = rho*kT - (T_ref/T)*(rho*kT_ref)

    clamping = max(0.0, bracket_term)

    term2 = clamping^nu_div_gamma

    return term1 * term2
end

function omega_0(rho::Float64, T::Float64, xi::Float64, constants::constants_k_H2)
    (; T_c, rho_c, P_CRIT, A1, A2, B1, B2, C1, C2, C3, R_D, ν, γ_crit, xi_0, GAMMA_0, qD, T_ref, k_B) = constants

    rhoc_div_rho = rho_c / rho
    denom = ( (qD*xi)^(-1.0) ) + ( ( ( qD*xi*rhoc_div_rho )^2.0 )/3.0 )
    exponent_term = -(1.0/denom)
    
    return (2.0/pi) * (1.0 - exp(exponent_term))
end

function omega(rho::Float64, T::Float64, xi::Float64, cp::Float64, cv::Float64, constants::constants_k_H2)
    (; T_c, rho_c, P_CRIT, A1, A2, B1, B2, C1, C2, C3, R_D, ν, γ_crit, xi_0, GAMMA_0, qD, T_ref, k_B) = constants

    term1 = ( (cp-cv)/cp ) * atan(qD*xi)

    term2 = (cv/cp) * qD * xi

    exponent_term = term1 + term2

    return (2.0/pi) * (exponent_term)
end

#The tricky one!
function delta_lambda_c(rho::Float64, T::Float64, cp::Float64, cv::Float64, kT::Float64, 
    kT_ref::Float64, nu_bar::Float64, constants::constants_k_H2)

    (; T_c, rho_c, P_CRIT, A1, A2, B1, B2, C1, C2, C3, R_D, ν, γ_crit, xi_0, GAMMA_0, qD, T_ref, k_B) = constants

    tol=1.0e-12
    xi_val = xi(rho, T, kT, kT_ref, constants)

    if (xi_val < tol)
        return 0.0
    end

    omega_0_val = omega_0(rho, T, xi_val, constants)
    omega_val = omega(rho, T, xi_val, cp, cv, constants)

    numerator = rho*cp*R_D*k_B*T
    denominator = 6.0*pi*nu_bar*xi_val

    return (numerator/denominator)*(omega_val-omega_0_val)
end


function thermal_conductivity_H2(rho::Float64, T::Float64, cp::Float64, cv::Float64, kT::Float64, 
    kT_ref::Float64, nu_bar::Float64)

    constants = constants_k_H2(
        32.938, #T_c
        31.323, #rho_c
        1.2858e6, #P_CRIT
        [-1.245, 310.212, -331.004, 246.016, -65.781, 10.826, -0.519659, 0.0143979], #A1
        [1.42304e4, -1.93922e4,  1.58379e4, -4.81812e3, 7.28639e2, -3.57365e1,  1.00000e0], #A2
        [2.65975e-2, -1.33826e-3,  1.30219e-2, -5.67678e-3, -9.23380e-5], #B1
        [-1.21727e-3,  3.66663e-3,  3.88715e-3, -9.21055e-3,  4.00723e-3], #B2
        3.57e-4, -2.46e-2, 0.2, #C1, C2, C3
        1.01, #R_D
        0.63, #ν
        1.2415, #γ_crit
        1.5e-10, #xi_0
        0.052, #GAMMA_0
        1.0 / 5.0e-10, #qD
        49.7175, #T_ref
        1.380649e-23 #k_B
    )

    (; T_c, rho_c, P_CRIT, A1, A2, B1, B2, C1, C2, C3, R_D, ν, γ_crit, xi_0, GAMMA_0, qD, T_ref, k_B) = constants

    lambda_0_val = lambda0(T, constants)
    delta_lambda_val = delta_lambda(rho, T, constants)

    lambda_crit_val = 0.0

    if abs(T_c - T) < 7.0 # If it is close to critical point (within 7 K) - use complex function
        lambda_crit_val = delta_lambda_c(rho, T, cp, cv, kT, kT_ref, nu_bar, constants)
    else # Otherwise simpler function is good enough
        lambda_crit_val = delta_lambda_c_empirical(rho, T, constants)
    end

    return lambda_0_val + delta_lambda_val + lambda_crit_val
end