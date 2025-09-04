export thermal_conductivity_N2

###Refer to "Viscosity and Thermal Conductivity Equations for Nitrogen, Oxygen, Argon, and Air", 2004


struct constants_k_N2{T<:AbstractFloat,V<:AbstractVector}
    T_c::T
    rho_c::T
    P_CRIT::T
    M::T

    epsilon_k::T # epsilon/k
    sigma::T

    b::V

    N_L0::V
    t_L0::V

    N_LR::V
    t_LR::V
    d_LR::V
    l_LR::V

    R_D::T
    ν::T
    γ_crit::T
    xi_0::T
    GAMMA_0::T
    qD::T
    T_ref::T
    k_B::T
end
Adapt.@adapt_structure constants_k_N2


function mu_0_N2_supplementary(T::F, constants::constants_k_N2) where F <: AbstractFloat
    (; M, sigma, epsilon_k, b) = constants
    
    Tstar = T / epsilon_k
    ln_Tstar = log(Tstar)
    
    ln_sum = b[1] + b[2]*ln_Tstar + b[3]*ln_Tstar^2 + b[4]*ln_Tstar^3 + b[5]*ln_Tstar^4
    omega  = exp(ln_sum)
    
    return ( F(0.0266958) * sqrt(M * T) ) / ( (sigma)^F(2) * omega )
end


function lambda0_N2(T::F, constants::constants_k_N2) where F <: AbstractFloat
    (; T_c, N_L0, t_L0) = constants
    
    tau = T_c / T
    
    mu_0_val = mu_0_N2_supplementary(T, constants)

    term1 = N_L0[1] * (mu_0_val)
    term2 = N_L0[2] * (tau^t_L0[2])
    term3 = N_L0[3] * (tau^t_L0[3])
    
    return term1 + term2 + term3
end

function lambda_r_N2(rho::F, T::F, constants::constants_k_N2, config) where F <: AbstractFloat
    (; T_c, rho_c, N_LR, t_LR, d_LR, l_LR) = constants

    backend = config.hardware.backend
    workgroup = config.hardware.workgroup

    tau = T_c / T
    delta = rho / rho_c

    term_sum = zero(F)


    ndrange = length(N_LR)
    kernel! = _lambda_r_N2(_setup(backend, workgroup, ndrange)...)
    kernel!(N_LR, tau, delta, t_LR, d_LR, l_LR, term, term_sum)

    # for i in eachindex(N_LR) # KERNEL!!!!!!!
    #     term = N_LR[i] * (tau^t_LR[i]) * (delta^d_LR[i])

    #     # The paper states that an exponential term is included only when its exponent l_i is not zero..
    #     if l_LR[i] != zero(F)
    #         term *= exp(-(delta^l_LR[i]))
    #     end

    #     term_sum += term
    # end
    return term_sum
end
@kernel inbounds=true function _lambda_r_N2(N_LR, tau, delta, t_LR, d_LR, l_LR, term, term_sum)
    i = @index(Global)

    term = N_LR[i] * (tau^t_LR[i]) * (delta^d_LR[i])

    if l_LR[i] != zero(F)
        term *= exp(-(delta^l_LR[i]))
    end

    term_sum += term
end






function xi(rho::F, T::F, kT::F, kT_ref::F, constants::constants_k_N2) where F <: AbstractFloat
    (; T_c, rho_c, P_CRIT, R_D, ν, γ_crit, xi_0, GAMMA_0, qD, T_ref, k_B, M) = constants
    # This piece of code is copied from Hydrogen file, it is just rearranged differently in the paper

    # kT = (1/rho) * (d rho / d p) at constant T
    # kT is evaluated at T passed into delta_lambda_c, while kT_ref evaluated at T_ref
    
    rho_c = rho_c * M

    nu_div_gamma = ν/γ_crit

    term1 = xi_0 * ((P_CRIT*rho)/(GAMMA_0*(rho_c^F(2))))^nu_div_gamma

    bracket_term = rho*kT - (T_ref/T)*(rho*kT_ref)

    clamping = max(zero(F), bracket_term)

    term2 = clamping^nu_div_gamma

    return term1 * term2
end

function omega_0(rho::F, T::F, xi::F, constants::constants_k_N2) where F <: AbstractFloat
    (; T_c, rho_c, P_CRIT, R_D, ν, γ_crit, xi_0, GAMMA_0, qD, T_ref, k_B, M) = constants

    rho_c = rho_c * M


    rhoc_div_rho = rho_c / rho
    xi_div_qD = xi / qD

    denom = ( (xi_div_qD)^(-one(F)) ) + ( ( ( xi_div_qD*rhoc_div_rho )^F(2) )/F(3) )

    exponent_term = (-one(F)/denom)

    return (F(2)/pi) * (one(F) - exp(exponent_term))
end

function omega(rho::F, T::F, xi::F, cp::F, cv::F, constants::constants_k_N2) where F <: AbstractFloat
    (; T_c, rho_c, P_CRIT, R_D, ν, γ_crit, xi_0, GAMMA_0, qD, T_ref, k_B) = constants

    xi_div_qD = xi / qD

    term1 = ( (cp-cv)/cp ) * atan(xi_div_qD)

    term2 = (cv/cp) * xi_div_qD

    exponent_term = term1 + term2

    return (F(2)/pi) * (exponent_term)
end

function lambda_c_N2(rho::F, T::F, cp::F, cv::F, kT::F, 
    kT_ref::F, nu_bar::F, constants::constants_k_N2) where F <: AbstractFloat

    cp = cp * F(1.0e3)
    cv = cv * F(1.0e3)
    nu_bar = nu_bar * F(1.0e-6)

    (; T_c, rho_c, P_CRIT, R_D, ν, γ_crit, xi_0, GAMMA_0, qD, T_ref, k_B) = constants

    tol=1.0e-12
    xi_val = xi(rho, T, kT, kT_ref, constants)

    if (xi_val < tol)
        return zero(F)
    end

    omega_0_val = omega_0(rho, T, xi_val, constants)
    omega_val = omega(rho, T, xi_val, cp, cv, constants)

    numerator = rho*cp*R_D*k_B*T
    denominator = F(6.0)*pi*nu_bar*xi_val

    return (numerator/denominator)*(omega_val-omega_0_val)
end




function thermal_conductivity_N2(rho::F, T::F, cp::F, cv::F, kT::F, 
    kT_ref::F, nu_bar::F, config) where F <: AbstractFloat

    constants = constants_k_N2(
        F(126.192),     # T_c (K)
        F(11.1839),     # rho_c (mol/dm^3)
        F(3.3958e6),    # P_CRIT (Pa)
        F(28.01348),    # M (g/mol)
        F(98.94),       # epsilon_k (K)
        F(0.3656),      # sigma (nm)
        [F(0.431), F(-0.4623), F(0.08406), F(0.005341), F(-0.00331)],             # b_i coefficients
        [F(1.511), F(2.117), F(-3.332)],                                          # N_L0 (i=1 to 3)
        [F(0.0), F(-1.0), F(-0.7)],                                               # t_L0 (i=1 to 3, t1 is not used)
        [F(8.862), F(31.11), F(-73.13), F(20.03), F(-0.7096), F(0.2672)],         # N_LR (i=4 to 9)
        [F(0.0), F(0.03), F(0.2), F(0.8), F(0.6), F(1.9)],                         # t_LR (i=4 to 9)
        [F(1.0), F(2.0), F(3.0), F(4.0), F(8.0), F(10.0)],                         # d_LR (i=4 to 9)
        [F(0.0), F(0.0), F(1.0), F(2.0), F(2.0), F(2.0)],                          # l_LR (i=4 to 9)
        F(1.01),        # R_D (R_0 in paper)
        F(0.63),        # nu
        F(1.2415),      # gamma_crit
        F(0.17e-9),     # xi_0
        F(0.055),       # GAMMA_0
        F(0.40e-9),     # qD
        F(252.384),     # T_ref (K)
        F(1.380658e-23) # k_B
    )


    rho_molar = rho / constants.M

    lambda_0_val = lambda0_N2(T, constants)
    lambda_r_val = lambda_r_N2(rho_molar, T, constants, config)
    
    lambda_crit_val = lambda_c_N2(rho, T, cp, cv, kT, kT_ref, nu_bar, constants)


    thermal_conductivity = lambda_0_val + lambda_r_val + lambda_crit_val

    return thermal_conductivity / F(1000.0) # Convert mW into W
end