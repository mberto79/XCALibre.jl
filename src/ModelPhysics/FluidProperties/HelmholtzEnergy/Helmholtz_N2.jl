using XCALibre

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



function alpha_0(δ::F, τ::F, constants, fluid::N2) where F <: AbstractFloat
    (; a_coeffs) = constants
    summations = ( a_coeffs[4] * (τ^(-one(F))) ) + ( a_coeffs[5] * (τ^(-F(2))) ) + ( a_coeffs[6] * (τ^(-F(3))) )

    last_term = a_coeffs[7] * log(one(F) - exp(-a_coeffs[8]*τ))

    return log(δ) + ( a_coeffs[1] * log(τ) ) + a_coeffs[2] + ( a_coeffs[3] * τ ) + summations + last_term
end


function d_alpha_0_d_delta(δ::F, τ::F, constants, fluid::N2) where F <: AbstractFloat
    return one(F) / δ
end

function d2_alpha_0_d_delta2(δ::F, τ::F, constants, fluid::N2) where F <: AbstractFloat
    return -one(F) / (δ^2)
end


function d_alpha_0_d_tau(δ::F, τ::F, constants, fluid::N2) where F <: AbstractFloat
    (; a_coeffs) = constants
    last_term = (a_coeffs[7]*a_coeffs[8])/(exp(a_coeffs[8]*τ)-one(F))
    return (a_coeffs[1]/τ) + a_coeffs[3] - (a_coeffs[4]/(τ^2)) - ((F(2)*a_coeffs[5])/(τ^3)) - ((F(3)*a_coeffs[6])/(τ^4)) + last_term
end

function d2_alpha_0_d_tau2(δ::F, τ::F, constants, fluid::N2) where F <: AbstractFloat
    (; a_coeffs) = constants
    exponent_term = exp(a_coeffs[8]*τ)
    last_term = (a_coeffs[7]*(a_coeffs[8]^2)*exponent_term)/((exponent_term-one(F))^2)
    return -(a_coeffs[1]/(τ^2)) + ((F(2)*a_coeffs[4])/(τ^3)) + ((F(6)*a_coeffs[5])/(τ^4)) + ((F(12)*a_coeffs[6])/(τ^5)) - last_term
end





### Residual Part
function alpha_r(δ::F, τ::F, constants, fluid::N2) where F <: AbstractFloat
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


function d_alpha_r_d_tau(δ::F, τ::F, constants, fluid::N2) where F <: AbstractFloat
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

function d_alpha_r_d_delta(δ::F, τ::F, constants, fluid::N2) where F <: AbstractFloat
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


function d2_alpha_r_d_tau2(δ::F, τ::F, constants, fluid::N2) where F <: AbstractFloat
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

function d2_alpha_r_d_delta2(δ::F, τ::F, constants, fluid::N2) where F <: AbstractFloat
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


function d2_alpha_r_d_delta_d_tau(δ::F, τ::F, constants, fluid::N2) where F <: AbstractFloat
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




function EOS_wrapper_N2(fluid::HelmholtzEnergyFluid, T::F, pressure::F) where F <: AbstractFloat
    constants = HelmholtzFluidConstants(
        F(126.192),     # T_c
        F(11.1839e3),   # rho_c, multiplied by e3 for convenience
        F(8.314472),    # R_univ
        F(28.01348e-3), # M_N2, multiplied by e-3 for convenience
        F(252.384),     # T_ref = T_c * 2
        
        F[2.5, -12.76952708, -0.00784163, -1.934819e-4, -1.247742e-5,
        6.678326e-8, 1.012941, 26.65788], # a_coeffs
        
        [zero(F), zero(F)], #dummy vector for k_coeffs

        F[ # Polynomial Part (k=1 to 6)
        0.924803575275, -0.492448489428, 0.661883336938, -0.192902649201e1, 
        -0.622469309629e-1, 0.349943957581,
        # Exponential Part (k=7 to 32)
        0.564857472498, -0.161720005987e1, -0.481395031883, 0.421150636384,
        -0.161962230825e-1, 0.172100994165, 0.735448924933e-2, 0.168077305479e-1,
        -0.107626664179e-2, -0.137318088513e-1, 0.635466899859e-3, 0.304432279419e-2,
        -0.4357623366045e-1, -0.723174889316e-1, 0.389644315272e-1, -0.212201363910e-1,
        0.40882298181509e-2, -0.551990017984e-4, -0.462016716479e-1, -0.300311716011e-2,
        0.368825891208e-1, -0.255856846220e-2, 0.896915264558e-2, -0.441513370350e-2,
        0.133722924858e-2, 0.264832491957e-3,
        # Gaussian Part (k=33 to 36)
        0.19668894015e2, -0.209115600730e2, 0.1677883066989e-1, 0.2627675665274e4
        ], # N
        
        [F(0.25), F(0.875), F(0.5), F(0.875), F(0.375), F(0.75), F(0.5), F(0.75), F(2.0), F(1.25), F(3.5), 
        one(F), F(0.5), F(3.0), zero(F), F(2.75), F(0.75), F(2.5), F(4.0), F(6.0), F(6.0), F(3.0), F(3.0), F(6.0), 
        F(16.0), F(11.0), F(15.0), F(12.0), F(12.0), F(7.0), F(4.0), F(16.0), zero(F), one(F), F(2.0), F(3.0)], # t

        [one(F), one(F), F(2.0), F(2.0), F(3.0), F(3.0), one(F), one(F), one(F), F(3.0), F(3.0), F(4.0), F(6.0), 
        F(6.0), F(7.0), F(7.0), F(8.0), F(8.0), one(F), F(2.0), F(3.0), F(4.0), F(5.0), F(8.0), F(4.0), F(5.0), 
        F(5.0), F(8.0), F(3.0), F(5.0), F(6.0), F(9.0), one(F), one(F), F(3.0), F(2.0)], # d

        F[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 0, 0, 0, 0], # p

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
        
        F[-6.12445284, 1.26327220, -0.765910082, -1.77570564], # vapour_N
        
        [one(F), F(1.5), F(2.5), F(5.0)], # vapour_k

        F(63.151),   # T_triple
        F(12.528e3), # p_triple
        F(5.0)       # Fluid dependent density guess multiplier to get liquid function
    )

    EOS_wrapper(fluid, T, pressure, constants)

end