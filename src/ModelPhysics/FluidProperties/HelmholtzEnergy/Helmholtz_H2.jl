### PREPARATORY FUNCTIONS


function alpha_0(δ::F, τ::F, constants, fluid::Union{H2,H2_para}) where F <: AbstractFloat
    (; T_c, a_coeffs, k_coeffs) = constants
    sum_val = zero(F)
    for i in 4:lastindex(a_coeffs)
        sum_val += a_coeffs[i] * log(one(F) - exp((-k_coeffs[i] * τ)/T_c))
    end
    return log(δ) + ( (a_coeffs[1] - one(F)) * log(τ) ) + a_coeffs[2] + ( a_coeffs[3] * τ ) + sum_val
end

function d_alpha_0_d_delta(δ::F, τ::F, constants, fluid::Union{H2,H2_para}) where F <: AbstractFloat
    return one(F) / δ
end

function d2_alpha_0_d_delta2(δ::F, τ::F, constants, fluid::Union{H2,H2_para}) where F <: AbstractFloat
    return -one(F) / (δ^2)
end

function d_alpha_0_d_tau(δ::F, τ::F, constants, fluid::Union{H2,H2_para}) where F <: AbstractFloat
    (; T_c, a_coeffs, k_coeffs) = constants
    sum_val = zero(F)
    for i in 4:lastindex(a_coeffs)
        sum_val += (a_coeffs[i] * k_coeffs[i])/(T_c*(exp( (k_coeffs[i]*τ)/T_c ) - one(F)))
    end
    return ((a_coeffs[1] - one(F))/τ) + a_coeffs[3] + sum_val
end

function d2_alpha_0_d_tau2(δ::F, τ::F, constants, fluid::Union{H2,H2_para}) where F <: AbstractFloat
    (; T_c, a_coeffs, k_coeffs) = constants
    sum_val = zero(F)
    for i in 4:lastindex(a_coeffs)
        sum_val += (a_coeffs[i] * ( k_coeffs[i]/T_c )^2) * (exp( (k_coeffs[i]*τ)/T_c )/((exp( (k_coeffs[i]*τ)/T_c ) - one(F))^2))
    end
    return -((a_coeffs[1] - one(F))/(τ^2)) - sum_val
end


### Residual Part
function alpha_r(δ::F, τ::F, constants, fluid::Union{H2,H2_para}) where F <: AbstractFloat
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


function d_alpha_r_d_tau(δ::F, τ::F, constants, fluid::Union{H2,H2_para}) where F <: AbstractFloat
    (; N, d, t, p, α, β, γ, D) = constants
    term1 = zero(F)
    term2 = zero(F)
    term3 = zero(F)

    for i in 1:7
        if t[i] != zero(F)
            term1 += N[i] * (δ^d[i]) * t[i] * (τ^(t[i]-one(F)))
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

function d_alpha_r_d_delta(δ::F, τ::F, constants, fluid::Union{H2,H2_para}) where F <: AbstractFloat
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


function d2_alpha_r_d_tau2(δ::F, τ::F, constants, fluid::Union{H2,H2_para}) where F <: AbstractFloat
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

function d2_alpha_r_d_delta2(δ::F, τ::F, constants, fluid::Union{H2,H2_para}) where F <: AbstractFloat
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


function d2_alpha_r_d_delta_d_tau(δ::F, τ::F, constants, fluid::Union{H2,H2_para}) where F <: AbstractFloat
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


### NORMAL HYDROGEN

function EOS_wrapper_H2(fluid::H2, T::F, pressure::F) where F <: AbstractFloat
    constants = HelmholtzFluidConstants(
        F(33.145), # T_c (K)
        F(15.508e3), # rho_c (mol/m^3)
        F(8.314472), # R_univ (J/mol K)
        F(2.01588e-3), # M_H2
        F(49.7175),     # T_ref

                #Ideal-gas coefficients from Tables 3 & 4

        F[2.5, -1.4579856475, 1.888076782, 1.616, -0.4117, -0.792, 0.758, 1.217], # a_coeffs
        [zero(F), zero(F), zero(F), F(531.0), F(751.0), F(1989.0), F(2484.0), F(6859.0)], # k_coeffs

            # Residual Helmholtz energy coefficients from Table 5

        F[-6.93643, 0.01, 2.1101, 4.52059, 0.732564, -1.34086, 0.130985, -0.777414,
            0.351944, -0.0211716, 0.0226312, 0.032187, -0.0231752, 0.0557346], # N

        [F(0.6844), one(F), F(0.989), F(0.489), F(0.803), F(1.1444), F(1.409), F(1.754),
            F(1.311), F(4.187), F(5.646), F(0.791), F(7.249), F(2.986)], # t

        F[1.0, 4.0, 1.0, 1.0, 2.0, 2.0, 3.0, 1.0, 3.0, 2.0, 1.0, 3.0, 1.0, 1.0], # d

        F[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], # p

            # Gaussian term parameters from Table 6

        [zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F),
        F(-1.685), F(-0.489), F(-0.103), F(-2.506), F(-1.607)], # α

        [zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F),
        F(-0.171), F(-0.2245), F(-0.1304), F(-0.2785), F(-0.3967)], # β

        [zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F),
        (0.7164), F(1.3444), F(1.4517), F(0.7204), F(1.5445)], # γ

        [zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F),
        F(1.506), F(0.156), F(1.736), F(0.67), F(1.662)], # D

        F(1.2964e6), # p_c (Pa)

            #Vapour pressure ancillary equation coefficients from Table 8

        F[-4.89789, 0.988558, 0.349689, 0.499356], # vapour_N

        [one(F), F(1.5), F(2.0), F(2.85)], # vapour_k

            # Triple point properties from Table 7

        F(13.957), # T_triple (K)
        F(7360.0), # p_triple (Pa)
        F(2.5) # Fluid dependent density guess multiplier to get liquid function
    )

    EOS_wrapper(fluid, T, pressure, constants)
    
end



### PARAHYDROGEN

function EOS_wrapper_H2(fluid::H2_para, T::F, pressure::F) where F <: AbstractFloat

    constants = HelmholtzFluidConstants(
        F(32.938),      # T_c
        F(15.538e3),    # rho_c, multiplied by e3 for convenience
        F(8.314472),    # R_univ
        F(2.01588e-3),  # M_H2, multiplied by e-3 for convenience
        F(49.7175),     # T_ref
        
        F[2.5, -1.4485891134, 1.884521239, 4.30256, 13.0289,
        -47.7365, 50.0013, -18.6261, 0.993973, 0.536078], # a_coeffs

        [zero(F), zero(F), zero(F), F(499.0), F(826.5), F(970.8), F(1166.2), F(1341.4), F(5395.0), F(10185.0)], # k_coeffs

        F[-7.33375, 0.01, 2.60375, 4.66279, 0.68239, -1.47078, 0.135801,
        -1.05327, 0.328239, -0.0577833, 0.0449743, 0.0703464, -0.0401766, 0.11951], # N

        [F(0.6855), one(F), one(F), F(0.489), F(0.774), F(1.133), F(1.386), F(1.619), F(1.162), F(3.96),
        F(5.276), F(0.99), F(6.791), F(3.19)], # t

        F[1.0, 4.0, 1.0, 1.0, 2.0, 2.0, 3.0, 1.0, 3.0, 2.0, 1.0, 3.0, 1.0, 1.0], # d

        F[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], # p
        
        [zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F),
        F(-1.7437), F(-0.5516), F(-0.0634), F(-2.1341), F(-1.777)], # α

        [zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F),
        F(-0.194), F(-0.2019), F(-0.0301), F(-0.2383), F(-0.3253)], # β

        [zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F),
        F(0.8048), F(1.5248), F(0.6648), F(0.6832), F(1.493)], # γ

        [zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F), zero(F),
        F(1.5487), F(0.1785), F(1.28), F(0.6319), F(1.7104)], # D

        F(1.2858e6), # p_c
        
        F[-4.87767, 1.03359, 0.82668, -0.129412], # vapour_N
        
        [one(F), F(1.5), F(2.65), F(7.4)], # vapour_k
        
        F(13.8033), # T_triple
        F(7042.0),  # p_triple
        F(2.5)      # Fluid dependent density guess multiplier to get liquid function
    )

    EOS_wrapper(fluid, T, pressure, constants)
    
end