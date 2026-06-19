export mu_high_fidelity_H2

###Refer to "Correlation for the Viscosity of Normal Hydrogen Obtained from Symbolic Regression", 2013

struct constants_mu_H2{T<:AbstractFloat,V<:AbstractVector}
    M::T
    sigma::T
    epsilon_div_kb::T
    T_c::T
    rho_sc::T
    a::V
    b::V
    c::V
end
Adapt.@adapt_structure constants_mu_H2



function mu_0(T::F, constants::constants_mu_H2) where F <: AbstractFloat # so-called 'Zero-density Viscosity'
    (; T_c, M, sigma, epsilon_div_kb, rho_sc, a, b) = constants
    
    Tstar = T / epsilon_div_kb
    ln_Tstar = log(Tstar)
    
    ln_sum = a[1] + a[2]*ln_Tstar + a[3]*ln_Tstar^2 + a[4]*ln_Tstar^3 + a[5]*ln_Tstar^4
    S_starT_star = exp(ln_sum)
    
    return ( F(0.021357) * sqrt(M * T) ) / ( (sigma)^2 * S_starT_star )
end


function beta_mu(T::F, constants::constants_mu_H2) where F <: AbstractFloat # 'Second viscosity viral coefficient'
    (; T_c, M, sigma, epsilon_div_kb, rho_sc, a, b) = constants

    Tstar = T / epsilon_div_kb
    
    Bstar = zero(F)
    for i in eachindex(b)
        Bstar += b[i] * (Tstar^( -(i-1) )) #typo in article, fixed here
    end
    
    return ((sigma^3)/3) * Bstar #typo in article, fixed here
end


function mu_1(T::F, constants::constants_mu_H2) where F <: AbstractFloat # 'The initial-density coefficient of viscosity'
    return mu_0(T, constants) * beta_mu(T, constants)
end




function mu_high_fidelity_H2(T::F, rho::F) where F <: AbstractFloat # End result
    constants = constants_mu_H2(
        F(2.01588),   # M
        F(0.297),     # sigma
        F(30.41),     # epsilon_div_kb
        F(33.145),    # T_c
        F(90.5),      # rho_sc
        F[0.209630, -0.455274, 0.143602, -0.0335325, 0.00276981], # a
        F[-0.1870, 2.4871, 3.7151, -11.0972, 9.0965, -3.8292, 0.5166], # b
        F[6.43449673, 4.56334068e-2, 0.232797868, 0.958326120,
        0.127941189, 0.363576595] # c
    )

    (; T_c, M, sigma, epsilon_div_kb, rho_sc, a, b, c) = constants

    T_r  = T / T_c
    rho_r  = rho / rho_sc

    exp_term = exp( c[2]*T_r + (c[3]/(T_r)) + ((c[4]*rho_r^2)/(c[5]+T_r)) + c[6]*(rho_r^6))

    return mu_0(T, constants) + ( mu_1(T, constants) * rho ) + ( c[1] * (rho_r^2) ) * exp_term
end