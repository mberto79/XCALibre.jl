export EOS_wrapper_H2

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


struct constants_EoS_H2{T<:AbstractFloat,V<:AbstractVector}
    T_c::T
    rho_c::T
    R_univ::T
    M::T
    T_ref::T
    a_para::V
    k_para::V
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
Adapt.@adapt_structure constants_EoS_H2




### PREPARATORY FUNCTIONS


function alpha_0(δ::F, τ::F, constants::constants_EoS_H2) where F <: AbstractFloat
    (; T_c, a_para, k_para) = constants
    sum_val = zero(F)
    for i in 4:10
        sum_val += a_para[i] * log(one(F) - exp((-k_para[i] * τ)/T_c))
    end
    return log(δ) + ( (a_para[1] - one(F)) * log(τ) ) + a_para[2] + ( a_para[3] * τ ) + sum_val
end

function d_alpha_0_d_delta(δ::F, τ::F, constants::constants_EoS_H2) where F <: AbstractFloat
    return one(F) / δ
end

function d2_alpha_0_d_delta2(δ::F, τ::F, constants::constants_EoS_H2) where F <: AbstractFloat
    return -one(F) / (δ^2)
end

function d_alpha_0_d_tau(δ::F, τ::F, constants::constants_EoS_H2) where F <: AbstractFloat
    (; T_c, a_para, k_para) = constants
    sum_val = zero(F)
    for i in 4:10
        sum_val += (a_para[i] * k_para[i])/(T_c*(exp( (k_para[i]*τ)/T_c ) - one(F)))
    end
    return ((a_para[1] - one(F))/τ) + a_para[3] + sum_val
end

function d2_alpha_0_d_tau2(δ::F, τ::F, constants::constants_EoS_H2) where F <: AbstractFloat
    (; T_c, a_para, k_para) = constants
    sum_val = zero(F)
    for i in 4:10
        sum_val += (a_para[i] * ( k_para[i]/T_c )^2) * (exp( (k_para[i]*τ)/T_c )/((exp( (k_para[i]*τ)/T_c ) - one(F))^2))
    end
    return -((a_para[1] - one(F))/(τ^2)) - sum_val
end


### Residual Part
function alpha_r(δ::F, τ::F, constants::constants_EoS_H2) where F <: AbstractFloat
    (; N, d, t, p, α, β, γ, D) = constants
    term1 = zero(F)
    term2 = zero(F)
    term3 = zero(F)

    for i in 1:7
        term1 +=  N[i] * (δ^d[i]) * (τ^t[i])
    end


    for i in 8:9
        term2 +=  N[i] * (δ^d[i]) * (τ^t[i]) * exp( -δ^p[i] )
    end


    for i in 10:14
        expo = α[i]*(δ - D[i])^2 + β[i]*(τ-γ[i])^2

        term3  += N[i] * (δ^d[i]) * (τ^t[i]) * exp(expo)
    end

    return term1 + term2 + term3
end


function d_alpha_r_d_tau(δ::F, τ::F, constants::constants_EoS_H2) where F <: AbstractFloat
    (; N, d, t, p, α, β, γ, D) = constants
    term1 = zero(F)
    term2 = zero(F)
    term3 = zero(F)

    for i in 1:7
        if t[i] != zero(F)
            term1 += N[i] * (δ^d[i]) * t[i] * (τ^(t[i]-1))
        end
    end

    for i in 8:9
        if t[i] != zero(F)
            term2 += N[i] * (δ^d[i]) * t[i] * (τ^(t[i]-1)) * exp(-δ^p[i])
        end
    end

    for i in 10:14
        expo = α[i]*(δ - D[i])^2 + β[i]*(τ - γ[i])^2
        term_i = N[i] * (δ^d[i]) * (τ^t[i]) * exp(expo)
        factor = (t[i] / τ) + F(2) * β[i] * (τ - γ[i])
        term3 += term_i * factor
    end

    return term1 + term2 + term3
end

function d_alpha_r_d_delta(δ::F, τ::F, constants::constants_EoS_H2) where F <: AbstractFloat
    (; N, d, t, p, α, β, γ, D) = constants
    term1 = zero(F)
    term2 = zero(F)
    term3 = zero(F)

    for i in 1:7
        if d[i] != zero(F)
            term1 += N[i] * d[i] * (δ^(d[i]-1)) * (τ^t[i])
        end
    end

    for i in 8:9
        term_i = N[i] * (δ^d[i]) * (τ^t[i]) * exp(-δ^p[i])
        factor = (d[i] / δ) - p[i] * (δ^(p[i]-one(F)))
        term2 += term_i * factor
    end

    for i in 10:14
        expo = α[i]*(δ - D[i])^2 + β[i]*(τ-γ[i])^2
        term_i = N[i] * (δ^d[i]) * (τ^t[i]) * exp(expo)
        factor = (d[i] / δ) + F(2) * α[i] * (δ - D[i])
        term3 += term_i * factor
    end

    return term1 + term2 + term3
end


function d2_alpha_r_d_tau2(δ::F, τ::F, constants::constants_EoS_H2) where F <: AbstractFloat
    (; N, d, t, p, α, β, γ, D) = constants
    term1 = zero(F)
    term2 = zero(F)
    term3 = zero(F)

    for i in 1:7
        term1 += N[i] * (δ^d[i]) * t[i] * (t[i]-1) * (τ^(t[i]-F(2)))
    end

    for i in 8:9
        term2 += N[i] * (δ^d[i]) * t[i] * (t[i]-one(F)) * (τ^(t[i]-F(2))) * exp(-δ^p[i])
    end

    for i in 10:14
        expo = α[i]*(δ - D[i])^2 + β[i]*(τ - γ[i])^2
        term_i = N[i] * (δ^d[i]) * (τ^t[i]) * exp(expo)
        
        A_i = (t[i] / τ) + F(2) * β[i] * (τ - γ[i])
        A_prime_i = -(t[i] / τ^2) + F(2) * β[i]
        
        term3 += term_i * (A_i^2 + A_prime_i)
    end

    return term1 + term2 + term3
end

function d2_alpha_r_d_delta2(δ::F, τ::F, constants::constants_EoS_H2) where F <: AbstractFloat
    (; N, d, t, p, α, β, γ, D) = constants
    term1 = zero(F)
    term2 = zero(F)
    term3 = zero(F)

    for i in 1:7
        term1 += N[i] * d[i] * (d[i]-one(F)) * (δ^(d[i]-2)) * (τ^t[i])
    end

    for i in 8:9
        term_i = N[i] * (δ^d[i]) * (τ^t[i]) * exp(-δ^p[i])
        B_i = (d[i] / δ) - p[i] * (δ^(p[i]-1))
        B_prime_i = -(d[i] / δ^2) - p[i] * (p[i]-one(F)) * (δ^(p[i]-F(2)))
        term2 += term_i * (B_i^2 + B_prime_i)
    end

    for i in 10:14
        expo = α[i]*(δ - D[i])^2 + β[i]*(τ - γ[i])^2
        term_i = N[i] * (δ^d[i]) * (τ^t[i]) * exp(expo)
        C_i = (d[i] / δ) + F(2) * α[i] * (δ - D[i])
        C_prime_i = -(d[i] / δ^2) + F(2) * α[i]
        term3 += term_i * (C_i^2 + C_prime_i)
    end

    return term1 + term2 + term3
end


function d2_alpha_r_d_delta_d_tau(δ::F, τ::F, constants::constants_EoS_H2) where F <: AbstractFloat
    (; N, d, t, p, α, β, γ, D) = constants
    term1 = zero(F)
    term2 = zero(F)
    term3 = zero(F)

    for i in 1:7
        term1 += N[i] * d[i] * t[i] * (δ^(d[i]-one(F))) * (τ^(t[i]-one(F)))
    end

    for i in 8:9
        term_i = N[i] * (δ^d[i]) * (τ^t[i]) * exp(-δ^p[i])
        B_i = (d[i] / δ) - p[i] * (δ^(p[i]-1))
        term2 += term_i * (t[i] / τ) * B_i
    end

    for i in 10:14
        expo = α[i]*(δ - D[i])^2 + β[i]*(τ - γ[i])^2
        term_i = N[i] * (δ^d[i]) * (τ^t[i]) * exp(expo)
        A_i = (t[i] / τ) + F(2) * β[i] * (τ - γ[i])
        C_i = (d[i] / δ) + F(2) * α[i] * (δ - D[i])
        term3 += term_i * A_i * C_i
    end

    return term1 + term2 + term3
end



function EOS_wrapper_H2(T::F, pressure::F) where F <: AbstractFloat

    constants = constants_EoS_H2(
        F(32.938),    # T_c
        F(15.538e3),  # rho_c, multiplied by e3 for convenience
        F(8.314472),  # R_univ
        F(2.01588e-3), # M_H2, multiplied by e-3 for convenience
        F(49.7175),   # T_ref
        
        [F(2.5), F(-1.4485891134), F(1.884521239), F(4.30256), F(13.0289),
        F(-47.7365), F(50.0013), F(-18.6261), F(0.993973), F(0.536078)], # a_para

        [zero(F), zero(F), zero(F), F(499.0), F(826.5), F(970.8), F(1166.2), F(1341.4), F(5395.0), F(10185.0)], # k_para

        [F(-7.33375), F(0.01), F(2.60375), F(4.66279), F(0.68239), F(-1.47078), F(0.135801),
        F(-1.05327), F(0.328239), F(-0.0577833), F(0.0449743), F(0.0703464), F(-0.0401766), F(0.11951)], # N

        [F(0.6855), one(F), one(F), F(0.489), F(0.774), F(1.133), F(1.386), F(1.619), F(1.162), F(3.96),
        F(5.276), F(0.99), F(6.791), F(3.19)], # t

        [F(1.0), F(4.0), F(1.0), F(1.0), F(2.0), F(2.0), F(3.0), F(1.0), F(3.0), F(2.0), F(1.0), F(3.0), F(1.0), F(1.0)], # d

        [F(0.0), F(0.0), F(0.0), F(0.0), F(0.0), F(0.0), F(0.0), F(1.0), F(1.0), F(0.0), F(0.0), F(0.0), F(0.0), F(0.0)], # p
        
        [zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F),
        F(-1.7437), F(-0.5516), F(-0.0634), F(-2.1341), F(-1.777)], # α

        [zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F),
        F(-0.194), F(-0.2019), F(-0.0301), F(-0.2383), F(-0.3253)], # β

        [zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F),
        F(0.8048), F(1.5248), F(0.6648), F(0.6832), F(1.493)], # γ

        [zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F),
        F(1.5487), F(0.1785), F(1.28), F(0.6319), F(1.7104)], # D

        F(1.2858e6), # p_c
        
        [F(-4.87767), F(1.03359), F(0.82668), F(-0.129412)], # vapour_N
        
        [one(F), F(1.5), F(2.65), F(7.4)], # vapour_k
        
        F(13.8033), # T_triple
        F(7042.0),  # p_triple
        F(2.5)      # Fluid dependent density guess multiplier to get liquid function
    )


    (; T_c, rho_c, R_univ, M, T_ref, a_para, k_para, N, t, d, p, α, β, γ, D, p_c, liquid_multiplier) = constants

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