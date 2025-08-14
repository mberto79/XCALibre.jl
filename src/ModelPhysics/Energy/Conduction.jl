export Conduction

"""
    Conduction <: AbstractEnergyModel

Type that represents energy conduction model for solids.

### Fields
- `T` Temperature scalar field on cells (ScalarField).
- `Tf` Temperature scalar field on faces (FaceScalarField).
"""
struct Conduction{S1,F1} <: AbstractEnergyModel
    T::S1
    Tf::F1
end
Adapt.@adapt_structure Conduction

#material::Symbol, rho::Float64
Energy{Conduction}() = begin # maybe assign rho based on the material?
    args = nothing
    ARGS = typeof(args)
    Energy{Conduction,ARGS}(args)
end

(energy::Energy{EnergyModel, ARG})(mesh, solid) where {EnergyModel<:Conduction,ARG} = begin
    T  = ScalarField(mesh)
    Tf = FaceScalarField(mesh)

    Conduction(T, Tf)
end


# Perhaps need to pass this as a list of arguments
function initialise(
    energy::Conduction, model::Physics{T1,F,SO,M,Tu,E,D,BI}, T_field, rDf, rhocp_field, k, kf, cp, rho, config
) where {T1,F,SO,M,Tu,E,D,BI}

    if typeof(model.solid) <: NonUniform
        k_vals, cp_vals = get_coefficients(model.solid.material, T_field)

        k.values .= k_vals
        cp.values .= cp_vals

        interpolate_harmonic!(kf, k, config)
        initialise!(rDf, 1.0)
        @. rDf.values *= (1.0/kf.values)

        @. rhocp_field.values = rho.values
        @. rhocp_field.values *= cp.values
    end

    return nothing
end




function energy!(
    energy::Conduction, model::Physics{T1,F,SO,M,Tu,E,D,BI}, T_field, rDf, rhocp_field, k, kf, cp, rho, config
) where {T1,F,SO,M,Tu,E,D,BI}
    k_vals, cp_vals = get_coefficients(model.solid.material, T_field)

    k.values .= k_vals
    cp.values .= cp_vals

    interpolate_harmonic!(kf, k, config)
    initialise!(rDf, 1.0)
    @. rDf.values *= (1.0/kf.values)

    @. rhocp_field.values = rho.values
    @. rhocp_field.values *= cp.values

    return nothing
end