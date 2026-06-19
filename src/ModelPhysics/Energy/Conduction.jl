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

Energy{Conduction}() = begin
    args = nothing
    ARGS = typeof(args)
    Energy{Conduction,ARGS}(args)
end

(energy::Energy{EnergyModel, ARG})(mesh, solid) where {EnergyModel<:Conduction,ARG} = begin
    T  = ScalarField(mesh)
    Tf = FaceScalarField(mesh)

    Conduction(T, Tf)
end


function initialise(
    energy::Conduction, model::Physics{T1,F,SO,M,Tu,E,D,BI}, T_field, rDf, rhocp_field, k, kf, cp, rho, config
) where {T1,F,SO,M,Tu,E,D,BI}

    if typeof(model.solid) <: NonUniform
        update_thermo_properties!(k, cp, T_field, model.solid)

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

    update_thermo_properties!(k, cp, T_field, model.solid)

    interpolate_harmonic!(kf, k, config)
    initialise!(rDf, 1.0)
    @. rDf.values *= (1.0/kf.values)

    @. rhocp_field.values = rho.values
    @. rhocp_field.values *= cp.values

    return nothing
end

# Note: use kernel in the next update
function update_thermo_properties!(k, cp, T, solid)
    (; k_coeffs, cp_coeffs) = solid

    @. k.values = k_coeffs(T.values)
    @. cp.values = cp_coeffs(T.values)

    nothing
end