export calculate_surface_tension

###Refer to "Recommended Correlations for the Surface Tension of Common Fluids", 2012

struct SurfaceTensionProperties{T<:AbstractFloat, V<:AbstractVector}
    T_c::T   # Critical temperature
    σ_coeffs::V
    n_coeffs::V
    T_min::T # Minimum valid temperature
    T_max::T # Maximum valid temperature
end
Adapt.@adapt_structure SurfaceTensionProperties

function get_surface_tension_properties(::H2)
    return SurfaceTensionProperties(33.145, [-1.4165, 0.746383, 0.675625], [0.63882, 0.659804, 0.619149], 14.86, 32.82)
end

function get_surface_tension_properties(::H2_para)
    return SurfaceTensionProperties(32.938, [0.005314], [1.060], 13.80, 31.00)
end

function get_surface_tension_properties(::N2)
    return SurfaceTensionProperties(126.192, [0.02898], [1.246], 64.80, 120.24)
end


function calculate_surface_tension(fluid::HelmholtzEnergyFluid, T::F) where F <: AbstractFloat
    properties = get_surface_tension_properties(fluid)

    if !(properties.T_min <= T <= properties.T_max)
        return zero(F)
    end
    
    reduced_temp_term = one(F) - (T / properties.T_c)
    
    surface_tension = zero(F)

    # Loop through all coefficients and sum the terms
    for i in eachindex(properties.σ_coeffs)
        σ_i = properties.σ_coeffs[i]
        n_i = properties.n_coeffs[i]
        surface_tension += σ_i * (reduced_temp_term^n_i)
    end

    return surface_tension
end