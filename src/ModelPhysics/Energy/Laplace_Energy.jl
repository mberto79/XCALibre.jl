export LaplaceEnergy

struct LaplaceEnergy{S1,F1} <: AbstractEnergyModel
    T::S1
    Tf::F1
end
Adapt.@adapt_structure LaplaceEnergy

# Energy API constructor: allow `Energy{LaplaceEnergy}()`

Energy{LaplaceEnergy}() = Energy{LaplaceEnergy,Nothing}(nothing)


Energy{LaplaceEnergy}() = begin
    args = nothing
    ARGS = typeof(args)
    Energy{LaplaceEnergy,ARGS}(args)
end

(energy::Energy{EnergyModel, ARG})(mesh, medium) where {EnergyModel<:LaplaceEnergy,ARG} = begin #solid instead of fluid???
    T  = ScalarField(mesh)
    Tf = FaceScalarField(mesh)
    LaplaceEnergy(T, Tf)
end


function initialise(
    energy::LaplaceEnergy, model::Physics{T1,ME,M,Tu,E,D,BI}, _config
) where {T1,ME,M,Tu,E,D,BI} #T?
    # nothing special to set up for pure Laplace
    return energy
end




function energy!(
    energy::LaplaceEnergy, model::Physics{T1,ME,M,Tu,E,D,BI}, config
) where {T1,ME,M,Tu,E,D,BI}
    return nothing
end
