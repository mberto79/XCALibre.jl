export thermal_conductivity_H2

#=
    Refer to "Correlation of the Thermal Conductivity of Normal and Parahydrogen
        from the Triple Point to 1000 K and up to 100MPa", 2011
=#


struct constants_k_H2{T<:AbstractFloat,V<:AbstractVector}
    T_c::T
    rho_c::T
    P_CRIT::T
    A1::V
    A2::V
    B1::V
    B2::V
    C1::T
    C2::T
    C3::T
    R_D::T
    ν::T
    γ_crit::T
    xi_0::T
    GAMMA_0::T
    qD::T
    T_ref::T
    k_B::T
end
Adapt.@adapt_structure constants_k_H2


function lambda0(T::F, constants::constants_k_H2) where F <: AbstractFloat
    (; T_c, A1, A2) = constants

    T_r = T / T_c
    # numerator = sum(A1[i+1] * (T_r)^i for i in 0:7)
    # denominator = sum(A2[i+1] *(T_r)^i for i in 0:6)
    numerator = sum(A1[i] * (T_r)^(i-1) for i in eachindex(A1))
    denominator = sum(A2[i] * (T_r)^(i-1) for i in eachindex(A2))
    return numerator / denominator
end


function delta_lambda(rho::F, T::F, constants::constants_k_H2) where F <: AbstractFloat
    (; T_c, rho_c, B1, B2) = constants

    T_r = T / T_c
    rho_r = rho / rho_c

    term_sum = zero(F)
    for i in 1:5
        term_sum += (B1[i] + (B2[i] * T_r) ) * (rho_r)^i
    end
    return term_sum
end


function delta_lambda_c_empirical(rho::F, T::F, constants::constants_k_H2) where F <: AbstractFloat #The easy version
    (; T_c, rho_c, C1, C2, C3) = constants

    delta_T_c   = (T / T_c) - one(F)
    delta_rho_c   = (rho / rho_c) - one(F)
    denominator  = C2 + abs(delta_T_c)

    if denominator <= zero(F)
        return zero(F)
    else
        return (C1 / denominator) * exp(-(C3 * delta_rho_c)^2)
    end
end

function xi(rho::F, T::F, kT::F, kT_ref::F, constants::constants_k_H2) where F <: AbstractFloat
    (; T_c, rho_c, P_CRIT, A1, A2, B1, B2, C1, C2, C3, R_D, ν, γ_crit, xi_0, GAMMA_0, qD, T_ref, k_B) = constants
    # kT = (1/rho) * (d rho / d p) at constant T
    # kT is evaluated at T passed into delta_lambda_c, while kT_ref evaluated at T_ref
    
    nu_div_gamma = ν/γ_crit

    term1 = xi_0 * ((P_CRIT*rho)/(GAMMA_0*(rho_c^2)))^nu_div_gamma

    bracket_term = rho*kT - (T_ref/T)*(rho*kT_ref)

    clamping = max(zero(F), bracket_term)

    term2 = clamping^nu_div_gamma

    return term1 * term2
end

function omega_0(rho::F, T::F, xi::F, constants::constants_k_H2) where F <: AbstractFloat
    (; T_c, rho_c, P_CRIT, A1, A2, B1, B2, C1, C2, C3, R_D, ν, γ_crit, xi_0, GAMMA_0, qD, T_ref, k_B) = constants

    rhoc_div_rho = rho_c / rho
    denom = ( (qD*xi)^(-one(F)) ) + ( ( ( qD*xi*rhoc_div_rho )^F(2) )/F(3) )
    exponent_term = -(one(F)/denom)
    
    return (F(2)/pi) * (one(F) - exp(exponent_term))
end

function omega(rho::F, T::F, xi::F, cp::F, cv::F, constants::constants_k_H2) where F <: AbstractFloat
    (; T_c, rho_c, P_CRIT, A1, A2, B1, B2, C1, C2, C3, R_D, ν, γ_crit, xi_0, GAMMA_0, qD, T_ref, k_B) = constants

    term1 = ( (cp-cv)/cp ) * atan(qD*xi)

    term2 = (cv/cp) * qD * xi

    exponent_term = term1 + term2

    return (F(2)/pi) * (exponent_term)
end

#The tricky one!
function delta_lambda_c(rho::F, T::F, cp::F, cv::F, kT::F, 
    kT_ref::F, nu_bar::F, constants::constants_k_H2) where F <: AbstractFloat

    (; T_c, rho_c, P_CRIT, A1, A2, B1, B2, C1, C2, C3, R_D, ν, γ_crit, xi_0, GAMMA_0, qD, T_ref, k_B) = constants

    tol=1.0e-12
    xi_val = xi(rho, T, kT, kT_ref, constants)

    if (xi_val < tol)
        return zero(F)
    end

    omega_0_val = omega_0(rho, T, xi_val, constants)
    omega_val = omega(rho, T, xi_val, cp, cv, constants)

    numerator = rho*cp*R_D*k_B*T
    denominator = F(6)*pi*nu_bar*xi_val

    return (numerator/denominator)*(omega_val-omega_0_val)
end


function thermal_conductivity_H2(fluid::H2, rho::F, T::F, cp::F, cv::F, kT::F, 
    kT_ref::F, nu_bar::F) where F <: AbstractFloat

    constants = constants_k_H2(
        F(33.145), #T_c
        F(31.262), #rho_c
        F(1.2964e6), #P_CRIT
        F[-0.340976, 4.58820, -1.45080, 0.326394, 3.16939e-3, 1.90592e-4, -1.13900e-6], #A1
        F[138.497, -22.1878, 4.57151, 1.0], #A2
        F[3.63081e-2, -2.07629e-2, 3.14810e-2, -1.43097e-2, 1.74980e-3], #B1
        F[1.83370e-3, -8.86716e-3, 1.58260e-2, -1.06283e-2, 2.80673e-3], #B2
        F(6.24e-4), F(-2.58e-7), F(0.837), #C1, C2, C3
        F(1.01), #R_D
        F(0.63), #ν
        F(1.2415), #γ_crit
        F(1.5e-10), #xi_0
        F(0.052), #GAMMA_0
        F(1.0 / 4.0e-10), #qD
        F(49.7175), #T_ref
        F(1.380649e-23) #k_B
    )

    thermal_conductivity_H2_wrapper(rho, T, cp, cv, kT, kT_ref, nu_bar, constants)
end

function thermal_conductivity_H2(fluid::H2_para, rho::F, T::F, cp::F, cv::F, kT::F, 
    kT_ref::F, nu_bar::F) where F <: AbstractFloat

    constants = constants_k_H2(
        F(32.938), #T_c
        F(31.323), #rho_c
        F(1.2858e6), #P_CRIT
        F[-1.245, 310.212, -331.004, 246.016, -65.781, 10.826, -0.519659, 0.0143979], #A1
        F[1.42304e4, -1.93922e4, 1.58379e4, -4.81812e3, 7.28639e2, -3.57365e1, 1.0], #A2
        F[2.65975e-2, -1.33826e-3, 1.30219e-2, -5.67678e-3, -9.23380e-5], #B1
        F[-1.21727e-3, 3.66663e-3, 3.88715e-3, -9.21055e-3, 4.00723e-3], #B2
        F(3.57e-4), F(-2.46e-2), F(0.2), #C1, C2, C3
        F(1.01), #R_D
        F(0.63), #ν
        F(1.2415), #γ_crit
        F(1.5e-10), #xi_0
        F(0.052), #GAMMA_0
        F(1.0 / 5.0e-10), #qD
        F(49.7175), #T_ref
        F(1.380649e-23) #k_B
    )

    thermal_conductivity_H2_wrapper(rho, T, cp, cv, kT, kT_ref, nu_bar, constants)
end



function thermal_conductivity_H2_wrapper(rho::F, T::F, cp::F, cv::F, kT::F, 
    kT_ref::F, nu_bar::F, constants) where F <: AbstractFloat

    (; T_c, rho_c, P_CRIT, A1, A2, B1, B2, C1, C2, C3, R_D, ν, γ_crit, xi_0, GAMMA_0, qD, T_ref, k_B) = constants

    lambda_0_val = lambda0(T, constants)
    delta_lambda_val = delta_lambda(rho, T, constants)

    lambda_crit_val = zero(F)

    # The paper mentioned that complex model must be used for the range 5-10 Kelvin near critical point
    # However, a parametric study revealed that there is no such thing as over-applying complex model, and it is just better to be used at all temps
    # lambda_crit_val = delta_lambda_c(rho, T, cp, cv, kT, kT_ref, nu_bar, constants)

    
    lambda_crit_val = delta_lambda_c(rho, T, cp, cv, kT, kT_ref, nu_bar, constants)

    return lambda_0_val + delta_lambda_val + lambda_crit_val

end

# Parametric study for model switcher (at 1 bar, improvements remain for other pressures too):

# [Temperature range to switch for complex function] error% at T=T_test
# [7 K]  8.443% at T=25.75 K
# [10 K] 6.408% at T=22.75 K
# [14 K] 2.388% at T=47.00 K
# [15 K] 2.181% at T=48.00 K
# [17 K] 1.847% at T=50.00 K
# [25 K] 1.082% at T=58.00 K
# [50 K] 0.359% at T=83.0 K
# [70 K] 0.216% at T=20.25 K
# [90 K] 0.216% at T=20.25 K

# [pure complex model] 0.216% at T=20.25 K