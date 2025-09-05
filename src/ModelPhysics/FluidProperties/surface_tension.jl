export calculate_surface_tension

###Refer to "Recommended Correlations for the Surface Tension of Common Fluids", 2012

struct SurfaceTensionProperties{T<:AbstractFloat}
    T_c::T   # Critical temperature
    σ_0::T
    n_0::T 
    T_min::T # Minimum valid temperature
    T_max::T # Maximum valid temperature
end
Adapt.@adapt_structure SurfaceTensionProperties

function get_surface_tension_properties(::H2)
    return SurfaceTensionProperties(32.938, 0.005314, 1.060, 13.80, 31.00)
end

function get_surface_tension_properties(::N2)
    return SurfaceTensionProperties(126.192, 0.02898, 1.246, 64.80, 120.24)
end

function calculate_surface_tension(fluid::HelmholtzEnergyFluid, T::F) where F <: AbstractFloat
    properties = get_surface_tension_properties(fluid)

    if !(properties.T_min <= T <= properties.T_max)
        return zero(F)
    end
    
    reduced_temp_term = one(F) - (T / properties.T_c)
    
    surface_tension = properties.σ_0 * (reduced_temp_term^properties.n_0)

    return surface_tension
end