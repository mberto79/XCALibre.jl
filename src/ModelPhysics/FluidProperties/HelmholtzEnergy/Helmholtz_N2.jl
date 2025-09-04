export EOS_wrapper_N2

export alpha_0,
       d_alpha_0_d_delta,
       d2_alpha_0_d_delta2,
       d_alpha_0_d_tau,
       d2_alpha_0_d_tau2,
       alpha_r,
       d_alpha_r_d_tau,
       d_alpha_r_d_delta,
       d2_alpha_r_d_tau2,
       d2_alpha_r_d_delta2,
       d2_alpha_r_d_delta_d_tau


struct constants_EoS_N2{T<:AbstractFloat,V<:AbstractVector}
    T_c::T
    rho_c::T
    M::T
    R_univ::T
    T_ref::T
    a_nitro::V
    N::V
    d::V
    t::V
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
Adapt.@adapt_structure constants_EoS_N2

function alpha_0(δ::F, τ::F, constants::constants_EoS_N2) where F <: AbstractFloat
    (; a_nitro) = constants
    summations = ( a_nitro[4] * (τ^(-one(F))) ) + ( a_nitro[5] * (τ^(-F(2))) ) + ( a_nitro[6] * (τ^(-F(3))) )

    last_term = a_nitro[7] * log(one(F) - exp(-a_nitro[8]*τ))

    return log(δ) + ( a_nitro[1] * log(τ) ) + a_nitro[2] + ( a_nitro[3] * τ ) + summations + last_term
end


function d_alpha_0_d_delta(δ::F, τ::F, constants::constants_EoS_N2) where F <: AbstractFloat
    return one(F) / δ
end

function d2_alpha_0_d_delta2(δ::F, τ::F, constants::constants_EoS_N2) where F <: AbstractFloat
    return -one(F) / (δ^2)
end


function d_alpha_0_d_tau(δ::F, τ::F, constants::constants_EoS_N2) where F <: AbstractFloat
    (; a_nitro) = constants
    last_term = (a_nitro[7]*a_nitro[8])/(exp(a_nitro[8]*τ)-one(F))
    return (a_nitro[1]/τ) + a_nitro[3] - (a_nitro[4]/(τ^2)) - ((F(2)*a_nitro[5])/(τ^3)) - ((F(3)*a_nitro[6])/(τ^4)) + last_term
end

function d2_alpha_0_d_tau2(δ::F, τ::F, constants::constants_EoS_N2) where F <: AbstractFloat
    (; a_nitro) = constants
    exponent_term = exp(a_nitro[8]*τ)
    last_term = (a_nitro[7]*(a_nitro[8]^2)*exponent_term)/((exponent_term-one(F))^2)
    return -(a_nitro[1]/(τ^2)) + ((F(2)*a_nitro[4])/(τ^3)) + ((F(6)*a_nitro[5])/(τ^4)) + ((F(12)*a_nitro[6])/(τ^5)) - last_term
end





### Residual Part
function alpha_r(δ::F, τ::F, constants::constants_EoS_N2) where F <: AbstractFloat
    (; N, d, t, p, α, β, γ, D) = constants
    term1 = zero(F)
    term2 = zero(F)
    term3 = zero(F)

    for i in 1:6
        term1 +=  N[i] * (δ^d[i]) * (τ^t[i])
    end


    for i in 7:32
        term2 +=  N[i] * (δ^d[i]) * (τ^t[i]) * exp( -δ^p[i] )
    end


    for i in 33:36
        expo = -α[i]*(δ - D[i])^2 - β[i]*(τ-γ[i])^2

        term3  += N[i] * (δ^d[i]) * (τ^t[i]) * exp(expo)
    end

    return term1 + term2 + term3
end


function d_alpha_r_d_tau(δ::F, τ::F, constants::constants_EoS_N2) where F <: AbstractFloat
    (; N, d, t, p, α, β, γ, D) = constants
    term1 = zero(F)
    term2 = zero(F)
    term3 = zero(F)

    for i in 1:6
        if t[i] != zero(F)
            term1 += N[i] * (δ^d[i]) * t[i] * (τ^(t[i]-1))
        end
    end

    for i in 7:32
        if t[i] != zero(F)
            term2 += N[i] * (δ^d[i]) * t[i] * (τ^(t[i]-1)) * exp(-δ^p[i])
        end
    end

    for i in 33:36
        expo = -α[i]*(δ - D[i])^2 - β[i]*(τ - γ[i])^2
        term_i = N[i] * (δ^d[i]) * (τ^t[i]) * exp(expo)
        factor = (t[i] / τ) - F(2) * β[i] * (τ - γ[i])
        term3 += term_i * factor
    end

    return term1 + term2 + term3
end

function d_alpha_r_d_delta(δ::F, τ::F, constants::constants_EoS_N2) where F <: AbstractFloat
    (; N, d, t, p, α, β, γ, D) = constants
    term1 = zero(F)
    term2 = zero(F)
    term3 = zero(F)

    for i in 1:6
        if d[i] != zero(F)
            term1 += N[i] * d[i] * (δ^(d[i]-1)) * (τ^t[i])
        end
    end

    for i in 7:32
        term_i = N[i] * (δ^d[i]) * (τ^t[i]) * exp(-δ^p[i])
        factor = (d[i] / δ) - p[i] * (δ^(p[i]-1))
        term2 += term_i * factor
    end

    for i in 33:36
        expo = -α[i]*(δ - D[i])^2 - β[i]*(τ - γ[i])^2
        term_i = N[i] * (δ^d[i]) * (τ^t[i]) * exp(expo)
        factor = (d[i] / δ) - F(2) * α[i] * (δ - D[i])
        term3 += term_i * factor
    end

    return term1 + term2 + term3
end


function d2_alpha_r_d_tau2(δ::F, τ::F, constants::constants_EoS_N2) where F <: AbstractFloat
    (; N, d, t, p, α, β, γ, D) = constants
    term1 = zero(F)
    term2 = zero(F)
    term3 = zero(F)

    for i in 1:6
        term1 += N[i] * (δ^d[i]) * t[i] * (t[i]-one(F)) * (τ^(t[i]-F(2)))
    end

    for i in 7:32
        term2 += N[i] * (δ^d[i]) * t[i] * (t[i]-one(F)) * (τ^(t[i]-F(2))) * exp(-δ^p[i])
    end

    for i in 33:36
        expo = -α[i]*(δ - D[i])^2 - β[i]*(τ - γ[i])^2
        term_i = N[i] * (δ^d[i]) * (τ^t[i]) * exp(expo)
        
        A_i = (t[i] / τ) - F(2) * β[i] * (τ - γ[i])
        A_prime_i = -(t[i] / τ^2) - F(2) * β[i]
        
        term3 += term_i * (A_i^2 + A_prime_i)
    end

    return term1 + term2 + term3
end

function d2_alpha_r_d_delta2(δ::F, τ::F, constants::constants_EoS_N2) where F <: AbstractFloat
    (; N, d, t, p, α, β, γ, D) = constants
    term1 = zero(F)
    term2 = zero(F)
    term3 = zero(F)

    for i in 1:6
        term1 += N[i] * d[i] * (d[i]-one(F)) * (δ^(d[i]-F(2))) * (τ^t[i])
    end

    for i in 7:32
        term_i = N[i] * (δ^d[i]) * (τ^t[i]) * exp(-δ^p[i])
        B_i = (d[i] / δ) - p[i] * (δ^(p[i]-one(F)))
        B_prime_i = -(d[i] / δ^2) - p[i] * (p[i]-one(F)) * (δ^(p[i]-F(2)))
        term2 += term_i * (B_i^2 + B_prime_i)
    end

    for i in 33:36
        expo = -α[i]*(δ - D[i])^2 - β[i]*(τ - γ[i])^2
        term_i = N[i] * (δ^d[i]) * (τ^t[i]) * exp(expo)
        C_i = (d[i] / δ) - F(2) * α[i] * (δ - D[i])
        C_prime_i = -(d[i] / δ^2) - F(2) * α[i]
        term3 += term_i * (C_i^2 + C_prime_i)
    end

    return term1 + term2 + term3
end


function d2_alpha_r_d_delta_d_tau(δ::F, τ::F, constants::constants_EoS_N2) where F <: AbstractFloat
    (; N, d, t, p, α, β, γ, D) = constants
    term1 = zero(F)
    term2 = zero(F)
    term3 = zero(F)

    for i in 1:6
        term1 += N[i] * d[i] * t[i] * (δ^(d[i]-one(F))) * (τ^(t[i]-one(F)))
    end

    for i in 7:32
        term_i = N[i] * (δ^d[i]) * (τ^t[i]) * exp(-δ^p[i])
        B_i = (d[i] / δ) - p[i] * (δ^(p[i]-one(F)))
        term2 += term_i * (t[i] / τ) * B_i
    end

    for i in 33:36
        expo = -α[i]*(δ - D[i])^2 - β[i]*(τ - γ[i])^2
        term_i = N[i] * (δ^d[i]) * (τ^t[i]) * exp(expo)
        A_i = (t[i] / τ) - F(2) * β[i] * (τ - γ[i])
        C_i = (d[i] / δ) - F(2) * α[i] * (δ - D[i])
        term3 += term_i * A_i * C_i
    end

    return term1 + term2 + term3
end








function EOS_wrapper_H2(T::F, pressure::F, alpha::F) where F <: AbstractFloat

    constants = constants_EoS_H2(
        32.938, # T_c
        15.538e3, # rho_c, multiplied by e3 for convenience
        8.314472, # R_univ
        2.01588e-3, # M_H2, multiplied by e-3 for convenience
        49.7175, # T_ref
        
        [2.5, -1.4485891134, 1.884521239, 4.30256, 13.0289,
        -47.7365, 50.0013, -18.6261, 0.993973, 0.536078], # a_para

        [zero(F), zero(F), zero(F), 499.0, 826.5, 970.8, 1166.2, 1341.4, 5395.0, 10185.0], # k_para

        [-7.33375, 0.01, 2.60375, 4.66279, 0.68239, -1.47078, 0.135801,
        -1.05327, 0.328239, -0.0577833, 0.0449743, 0.0703464, -0.0401766, 0.11951], # N

        [0.6855, one(F), one(F), 0.489, 0.774, 1.133, 1.386, 1.619, 1.162, 3.96,
        5.276, 0.99, 6.791, 3.19], # t

        [1, 4, 1, 1, 2, 2, 3, 1, 3, 2, 1, 3, 1, 1], # d

        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], # p
        [zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F),

        -1.7437, -0.5516, -0.0634, -2.1341, -1.777], # α
        [zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F),

        -0.194, -0.2019, -0.0301, -0.2383, -0.3253], # β
        [zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F),
        0.8048, 1.5248, 0.6648, 0.6832, 1.493], # γ

        [zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F),
        1.5487, 0.1785, 1.28, 0.6319, 1.7104], # D

        1.2858e6, # p_c
        
        [-4.87767, 1.03359, 0.82668, -0.129412], # vapour_N
        
        [one(F), 1.5, 2.65, 7.4], # vapour_k
        
        13.8033, # T_triple
        7042.0, # p_triple
        2.5 # Fluid dependent density guess multiplier to get liquid function
    )

    (; T_c, rho_c, R_univ, M, T_ref, a_para, k_para, N, t, d, p, α, β, γ, D, p_c, liquid_multiplier) = constants

    pressure_tol = 1e-4
    alpha_tol = 1e-8

    rho_mol = zero(F)
    rho_list = [zero(F), zero(F)]

    latentHeat = zero(F)
    T_sat = zero(F)

    m_lv = zero(F)
    m_vl = zero(F)

    # Firstly, account for supercritical fluid state
    if (T >= T_c) && (pressure >= p_c)
        rho_guess = pressure / (R_univ * T) # Ideal gas guess
        rho_mol = find_density_advanced(T, pressure, rho_guess, constants)

    else # Else it is liquid/vapour
        (P_sat, T_sat, rho_l_sat, rho_v_sat) = find_saturation_properties(T, pressure, constants)

        if ( abs(pressure - P_sat) / P_sat ) < pressure_tol # TWO PHASE REGION, pressure matched saturation line

            rho_mol_liquid = find_density_advanced(T, pressure, rho_l_sat, constants) #maybe T_sat is better?
            rho_mol_vapour = find_density_advanced(T, pressure, rho_v_sat, constants) #maybe T_sat is better?

            rho_list[1] = rho_mol_liquid 
            rho_list[2] = rho_mol_vapour

        elseif pressure < P_sat # VAPOUR REGION
            rho_guess = pressure / (R_univ * T) # Ideal gas guess
            rho_guess_ = liquid_multiplier * rho_c # Higher density guess
            rho_mol = find_density_advanced(T, pressure, rho_guess, constants)
            rho_liquid = find_density_advanced(T_sat, pressure, rho_guess_, constants)

            rho_list[1] = rho_liquid
            rho_list[2] = rho_mol

            m_lv = c_τ_evap * alpha * rho_liquid * ( (T - T_sat)/T_sat ) # ASSUME alpha=1 is liquid

        elseif pressure > P_sat # LIQUID REGION
            # println("LIQUID")
            rho_guess = liquid_multiplier * rho_c # Higher density guess
            rho_guess_ = pressure / (R_univ * T) # Ideal gas guess
            rho_mol = find_density_advanced(T, pressure, rho_guess, constants)
            rho_vapour = find_density_advanced(T_sat, pressure, rho_guess_, constants)

            rho_list[1] = rho_mol
            rho_list[2] = rho_vapour

            m_vl = c_τ_cond * (one(F)-alpha) * rho_vapour * ( (T_sat - T)/T_sat )# ASSUME alpha=0 is liquid vapour
        end

        if isnan(rho_mol)
            error("Failed to find a valid density for the given state.")
        end
    end

    if T >= T_c
        # Comment out for now - we don't look into supercritical fluids for this project

        # rho_val, cp_val, cv_val, kT_val, kT_ref_val, 
        # internal_energy_val, enthalpy_val, entropy_val, beta_val = params_computation(rho_mol, T, constants)

        # return rho_val, cp_val, cv_val, kT_val, kT_ref_val, 
        #         internal_energy_val, enthalpy_val, entropy_val, beta_val, latentHeat, T_sat, m_lv, m_vl
    else
        rho_vals = [zero(F), zero(F)]
        cv_vals = [zero(F), zero(F)]
        cp_vals = [zero(F), zero(F)]
        kT_vals = [zero(F), zero(F)]
        kT_ref_vals = [zero(F), zero(F)]
        internal_energy_vals = [zero(F), zero(F)]
        enthalpy_vals = [zero(F), zero(F)]
        entropy_vals = [zero(F), zero(F)]
        beta_vals = [zero(F), zero(F)]

        for i in eachindex(rho_list)
            rho_vals[i], cp_vals[i], cv_vals[i], kT_vals[i], kT_ref_vals[i], 
            internal_energy_vals[i], enthalpy_vals[i], entropy_vals[i], beta_vals[i] = params_computation(rho_list[i], T, constants)
        end

        latentHeat = enthalpy_vals[2] - enthalpy_vals[1] #enthalpy_V - enthalpy_L

        return rho_vals, cp_vals, cv_vals, kT_vals, 
                kT_ref_vals, internal_energy_vals, enthalpy_vals, beta_vals, entropy_vals, latentHeat, T_sat, m_lv, m_vl

    end
end


function EOS_wrapper_N2(T::F, pressure::F) where F <: AbstractFloat
    
    constants = constants_EoS_N2(
        F(126.192),    # T_c
        F(11.1839e3),  # rho_c, multiplied by e3 for convenience
        F(28.01348e-3), # M_N2, multiplied by e-3 for convenience
        F(8.314472),   # R_univ
        F(252.384),    # T_ref = T_c * 2
        
        [F(2.5), F(-12.76952708), F(-0.00784163), F(-1.934819e-4), F(-1.247742e-5),
        F(6.678326e-8), F(1.012941), F(26.65788)], # a_nitro

        [ # Polynomial Part (k=1 to 6)
        F(0.924803575275), F(-0.492448489428), F(0.661883336938), F(-0.192902649201e1), 
        F(-0.622469309629e-1), F(0.349943957581),
        # Exponential Part (k=7 to 32)
        F(0.564857472498), F(-0.161720005987e1), F(-0.481395031883), F(0.421150636384),
        F(-0.161962230825e-1), F(0.172100994165), F(0.735448924933e-2), F(0.168077305479e-1),
        F(-0.107626664179e-2), F(-0.137318088513e-1), F(0.635466899859e-3), F(0.304432279419e-2),
        F(-0.4357623366045e-1), F(-0.723174889316e-1), F(0.389644315272e-1), F(-0.212201363910e-1),
        F(0.40882298181509e-2), F(-0.551990017984e-4), F(-0.462016716479e-1), F(-0.300311716011e-2),
        F(0.368825891208e-1), F(-0.255856846220e-2), F(0.896915264558e-2), F(-0.441513370350e-2),
        F(0.133722924858e-2), F(0.264832491957e-3),
        # Gaussian Part (k=33 to 36)
        F(0.19668894015e2), F(-0.209115600730e2), F(0.1677883066989e-1), F(0.2627675665274e4)
        ], # N

        [one(F), one(F), F(2.0), F(2.0), F(3.0), F(3.0), one(F), one(F), one(F), F(3.0), F(3.0), F(4.0), F(6.0), 
        F(6.0), F(7.0), F(7.0), F(8.0), F(8.0), one(F), F(2.0), F(3.0), F(4.0), F(5.0), F(8.0), F(4.0), F(5.0), 
        F(5.0), F(8.0), F(3.0), F(5.0), F(6.0), F(9.0), one(F), one(F), F(3.0), F(2.0)], # d

        [F(0.25), F(0.875), F(0.5), F(0.875), F(0.375), F(0.75), F(0.5), F(0.75), F(2.0), F(1.25), F(3.5), 
        one(F), F(0.5), F(3.0), zero(F), F(2.75), F(0.75), F(2.5), F(4.0), F(6.0), F(6.0), F(3.0), F(3.0), F(6.0), 
        F(16.0), F(11.0), F(15.0), F(12.0), F(12.0), F(7.0), F(4.0), F(16.0), zero(F), one(F), F(2.0), F(3.0)], # t

        [F(0), F(0), F(0), F(0), F(0), F(0), F(1), F(1), F(1), F(1), F(1), F(1), F(1), F(1), F(1), F(1), F(1), F(1),
        F(2), F(2), F(2), F(2), F(2), F(2), F(3), F(3), F(3), F(3), F(4), F(4), F(4), F(4), F(0), F(0), F(0), F(0)], # p

        [zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), 
        zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), 
        zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), F(20.0), F(20.0), F(15.0), F(25.0)], # α

        [zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), 
        zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), 
        zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), F(325.0), F(325.0), F(300.0), F(275.0)], # β

        [zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), 
        zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), 
        zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), F(1.16), F(1.16), F(1.13), F(1.25)], # γ

        [one(F), one(F), one(F), one(F), one(F), one(F), one(F), one(F), one(F), one(F), one(F), one(F), one(F), 
        one(F), one(F), one(F), one(F), one(F), one(F), one(F), one(F), one(F), one(F), one(F), one(F), one(F), 
        one(F), one(F), one(F), one(F), one(F), one(F), one(F), one(F), one(F), one(F)],  # D

        F(3.3958e6), # p_c
        
        [F(-6.12445284), F(1.26327220), F(-0.765910082), F(-1.77570564)], # vapour_N
        
        [one(F), F(1.5), F(2.5), F(5.0)], # vapour_k

        F(63.151),   # T_triple
        F(12.528e3), # p_triple
        F(5.0)       # Fluid dependent density guess multiplier to get liquid function
    )

    
    
    (; T_c, rho_c, R_univ, M, T_ref, p_c, liquid_multiplier) = constants

    pressure_tol = 1e-4
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

    # Firstly, account for supercritical fluid / superheated vapour state
    if T >= T_c
        rho_guess = pressure / (R_univ * T) # Ideal gas guess
        rho_mol = find_density_advanced(T, pressure, rho_guess, constants)
        rho_list = [rho_mol, rho_mol] # if T > T_crit, we want to return two identical densities

    else # Else it is liquid/vapour
        (P_sat, T_sat, rho_l_sat, rho_v_sat) = find_saturation_properties(T, pressure, constants)

        if ( abs(pressure - P_sat) / P_sat ) < pressure_tol # TWO PHASE REGION, pressure matched saturation line

            rho_mol_liquid = find_density_advanced(T, pressure, rho_l_sat, constants) #maybe T_sat is better?
            rho_mol_vapour = find_density_advanced(T, pressure, rho_v_sat, constants) #maybe T_sat is better?

            rho_list[1] = rho_mol_liquid 
            rho_list[2] = rho_mol_vapour

        elseif pressure < P_sat # VAPOUR REGION
            rho_guess = pressure / (R_univ * T) # Ideal gas guess
            rho_guess_ = liquid_multiplier * rho_c # Higher density guess
            rho_mol = find_density_advanced(T, pressure, rho_guess, constants)
            rho_liquid = find_density_advanced(T_sat, pressure, rho_guess_, constants)

            rho_list[1] = rho_liquid
            rho_list[2] = rho_mol

            # m_lv = c_τ_evap * alpha * rho_liquid * ( (T - T_sat)/T_sat ) # ASSUME alpha=1 is liquid

        elseif pressure > P_sat # LIQUID REGION
            # println("LIQUID")
            rho_guess = liquid_multiplier * rho_c # Higher density guess
            rho_guess_ = pressure / (R_univ * T) # Ideal gas guess
            rho_mol = find_density_advanced(T, pressure, rho_guess, constants)
            rho_vapour = find_density_advanced(T_sat, pressure, rho_guess_, constants)

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
        internal_energy_vals[i], enthalpy_vals[i], entropy_vals[i], beta_vals[i] = params_computation(rho_list[i], T, constants)
    end

    latentHeat = enthalpy_vals[2] - enthalpy_vals[1] #enthalpy_V - enthalpy_L

    return rho_vals, cp_vals, cv_vals, kT_vals, 
            kT_ref_vals, internal_energy_vals, enthalpy_vals, beta_vals, entropy_vals, latentHeat, T_sat#, m_lv, m_vl
end
