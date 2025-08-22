export thermal_conductivity_N2

###Refer to "Viscosity and Thermal Conductivity Equations for Nitrogen, Oxygen, Argon, and Air", 2004

using XCALibre
using Printf

struct constants_k_N2
    T_c::Float64
    rho_c::Float64
    P_CRIT::Float64
    M::Float64

    epsilon_k::Float64 # epsilon/k
    sigma::Float64

    b::Vector{Float64}

    N_L0::Vector{Float64}
    t_L0::Vector{Float64}

    N_LR::Vector{Float64}
    t_LR::Vector{Float64}
    d_LR::Vector{Float64}
    l_LR::Vector{Float64}

    R_D::Float64
    ν::Float64
    γ_crit::Float64
    xi_0::Float64
    GAMMA_0::Float64
    qD::Float64
    T_ref::Float64
    k_B::Float64
end



function mu_0_N2_supplementary(T::Float64, constants::constants_k_N2)
    (; M, sigma, epsilon_k, b) = constants
    
    Tstar = T / epsilon_k
    ln_Tstar = log(Tstar)
    
    ln_sum = b[1] + b[2]*ln_Tstar + b[3]*ln_Tstar^2 + b[4]*ln_Tstar^3 + b[5]*ln_Tstar^4
    omega  = exp(ln_sum)
    
    return ( 0.0266958 * sqrt(M * T) ) / ( (sigma)^2 * omega )
end


function lambda0_N2(T::Float64, constants::constants_k_N2)
    (; T_c, N_L0, t_L0) = constants
    
    tau = T_c / T
    
    mu_0_val = mu_0_N2_supplementary(T, constants)

    term1 = N_L0[1] * (mu_0_val)
    term2 = N_L0[2] * (tau^t_L0[2])
    term3 = N_L0[3] * (tau^t_L0[3])
    
    return term1 + term2 + term3
end

function lambda_r_N2(rho::Float64, T::Float64, constants::constants_k_N2)
    (; T_c, rho_c, N_LR, t_LR, d_LR, l_LR) = constants

    tau = T_c / T
    delta = rho / rho_c

    term_sum = 0.0
    for i in eachindex(N_LR)
        term = N_LR[i] * (tau^t_LR[i]) * (delta^d_LR[i])

        # The paper states that an exponential term is included only when its exponent l_i is not zero..
        if l_LR[i] != 0.0
            term *= exp(-(delta^l_LR[i]))
        end

        term_sum += term
    end
    return term_sum
end







function xi(rho::Float64, T::Float64, kT::Float64, kT_ref::Float64, constants::constants_k_N2)
    (; T_c, rho_c, P_CRIT, R_D, ν, γ_crit, xi_0, GAMMA_0, qD, T_ref, k_B, M) = constants
    # This piece of code is copied from Hydrogen file, it is just rearranged differently in the paper

    # kT = (1/rho) * (d rho / d p) at constant T
    # kT is evaluated at T passed into delta_lambda_c, while kT_ref evaluated at T_ref
    
    rho_c = rho_c * M

    nu_div_gamma = ν/γ_crit

    term1 = xi_0 * ((P_CRIT*rho)/(GAMMA_0*(rho_c^2)))^nu_div_gamma

    bracket_term = rho*kT - (T_ref/T)*(rho*kT_ref)

    clamping = max(0.0, bracket_term)

    term2 = clamping^nu_div_gamma

    return term1 * term2
end

function omega_0(rho::Float64, T::Float64, xi::Float64, constants::constants_k_N2)
    (; T_c, rho_c, P_CRIT, R_D, ν, γ_crit, xi_0, GAMMA_0, qD, T_ref, k_B, M) = constants

    rho_c = rho_c * M


    rhoc_div_rho = rho_c / rho
    xi_div_qD = xi / qD

    denom = ( (xi_div_qD)^(-1.0) ) + ( ( ( xi_div_qD*rhoc_div_rho )^2.0 )/3.0 )

    exponent_term = (-1.0/denom)

    return (2.0/pi) * (1.0 - exp(exponent_term))
end

function omega(rho::Float64, T::Float64, xi::Float64, cp::Float64, cv::Float64, constants::constants_k_N2)
    (; T_c, rho_c, P_CRIT, R_D, ν, γ_crit, xi_0, GAMMA_0, qD, T_ref, k_B) = constants

    xi_div_qD = xi / qD

    term1 = ( (cp-cv)/cp ) * atan(xi_div_qD)

    term2 = (cv/cp) * xi_div_qD

    exponent_term = term1 + term2

    return (2.0/pi) * (exponent_term)
end

function lambda_c_N2(rho::Float64, T::Float64, cp::Float64, cv::Float64, kT::Float64, 
    kT_ref::Float64, nu_bar::Float64, constants::constants_k_N2)

    cp = cp * 1.0e3
    cv = cv * 1.0e3
    nu_bar = nu_bar * 1.0e-6

    (; T_c, rho_c, P_CRIT, R_D, ν, γ_crit, xi_0, GAMMA_0, qD, T_ref, k_B) = constants

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




function thermal_conductivity_N2(rho::Float64, T::Float64, cp::Float64, cv::Float64, kT::Float64, 
    kT_ref::Float64, nu_bar::Float64)

    constants = constants_k_N2(
        126.192,     # T_c (K)
        11.1839,     # rho_c (mol/dm^3)
        3.3958e6,    # P_CRIT (Pa)
        28.01348,    # M (g/mol)
        98.94,       # epsilon_k (K)
        0.3656,   # sigma (nm)
        [0.431, -0.4623, 0.08406, 0.005341, -0.00331], # b_i coefficients
        [1.511, 2.117, -3.332], # N_L0 (i=1 to 3)
        [0.0, -1.0, -0.7],     # t_L0 (i=1 to 3, t1 is not used)
        [8.862, 31.11, -73.13, 20.03, -0.7096, 0.2672], # N_LR (i=4 to 9)
        [0.0, 0.03, 0.2, 0.8, 0.6, 1.9],               # t_LR (i=4 to 9)
        [1.0, 2.0, 3.0, 4.0, 8.0, 10.0],                # d_LR (i=4 to 9)
        [0.0, 0.0, 1.0, 2.0, 2.0, 2.0],                 # l_LR (i=4 to 9)
        1.01,        # R_D (R_0 in paper)
        0.63,        # nu
        1.2415,      # gamma_crit
        0.17e-9,     # xi_0
        0.055,       # GAMMA_0
        0.40e-9,     # qD
        252.384,     # T_ref (K)
        1.380658e-23 # k_B
    )

    rho_molar = rho / constants.M

    lambda_0_val = lambda0_N2(T, constants)
    lambda_r_val = lambda_r_N2(rho_molar, T, constants)
    
    lambda_crit_val = lambda_c_N2(rho, T, cp, cv, kT, kT_ref, nu_bar, constants)


    thermal_conductivity = lambda_0_val + lambda_r_val + lambda_crit_val

    return thermal_conductivity / 1000.0 # Convert mW into W
end