export mu_high_fidelity_N2

### NOTE: this function works also with Oxygen, Air, Argon; different coeffs are required though

###Refer to "Viscosity and Thermal Conductivity Equations for Nitrogen, Oxygen, Argon, and Air", 2004

# Constants for Nitrogen (N2)

struct constants_mu_N2{T<:AbstractFloat,V<:AbstractVector}
    M::T
    sigma::T
    epsilon_div_kb::T
    T_c::T
    rho_c::T
    b::V
    N_coeffs::V
    t_coeffs::V
    d_coeffs::V
    l_coeffs::V
end
Adapt.@adapt_structure constants_mu_N2


function mu_0_N2(T::F, constants::constants_mu_N2) where F <: AbstractFloat
    (; M, sigma, epsilon_div_kb, b) = constants
    Tstar = T / epsilon_div_kb
    ln_Tstar = log(Tstar)
    
    ln_sum = b[1] + b[2]*ln_Tstar + b[3]*ln_Tstar^2 + b[4]*ln_Tstar^3 + b[5]*ln_Tstar^4
    omega  = exp(ln_sum)
    
    return ( F(0.0266958) * sqrt(M * T) ) / ( (sigma)^2 * omega )
end

function mu_r(δ::F, τ::F, constants::constants_mu_N2) where F <: AbstractFloat
    (; N_coeffs, t_coeffs, d_coeffs, l_coeffs) = constants
    residual_viscosity = zero(F)

    for i in eachindex(N_coeffs)
        
        gamma = zero(F)
        
        if l_coeffs[i] == zero(F)
            # If the exponent l is zero, gamma is zero
            gamma = zero(F)
        else
            # If the exponent l is not zero, gamma is one
            gamma = one(F)
        end

        term = N_coeffs[i] * (τ^t_coeffs[i]) * (δ^d_coeffs[i]) * exp(-gamma * (δ^l_coeffs[i]))
        residual_viscosity += term
    end

    return residual_viscosity
end

function mu_high_fidelity_N2(T::F, rho::F) where F <: AbstractFloat

    constants = constants_mu_N2(
        F(28.01348),   # M
        F(0.3656),     # sigma
        F(98.94),      # epsilon_div_kb
        F(126.192),    # T_c
        F(11.1839),    # rho_c
        F[0.431, -0.4623, 0.08406, 0.005341, -0.00331],   # b
        F[10.72, 0.03989, 0.001208, -7.402, 4.620],       # N_coeffs
        F[0.1, 0.25, 3.2, 0.9, 0.3],                      # t_coeffs
        F[2, 10, 12, 2, 1],                               # d_coeffs
        F[0, 1, 1, 2, 3]                                  # l_coeffs
    )
    
    (; M, T_c, rho_c) = constants

    rho = rho / M

    τ = T_c / T
    δ = rho / rho_c

    return mu_0_N2(T, constants) + mu_r(δ, τ, constants)
end