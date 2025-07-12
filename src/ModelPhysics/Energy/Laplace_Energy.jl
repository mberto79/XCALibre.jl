export LaplaceEnergy

struct LaplaceEnergy{S1,F1,F2,S2,S3,F3,S4,S5} <: AbstractEnergyModel
    T::S1
    Tf::F1
    rDf::F2           
    rho::S2
    k::S3
    kf::F3
    cp::S4
    rhocp::S5
end
Adapt.@adapt_structure LaplaceEnergy

Energy{LaplaceEnergy}() = Energy{LaplaceEnergy,Nothing}(nothing)


Energy{LaplaceEnergy}() = begin
    args = nothing
    ARGS = typeof(args)
    Energy{LaplaceEnergy,ARGS}(args)
end

(energy::Energy{EnergyModel, ARG})(mesh, solid) where {EnergyModel<:LaplaceEnergy,ARG} = begin 
    T  = ScalarField(mesh)
    Tf = FaceScalarField(mesh)

    rDf = FaceScalarField(mesh)
    rho  = ScalarField(mesh)

    k  = ScalarField(mesh)
    kf = FaceScalarField(mesh)

    cp  = ScalarField(mesh)

    rhocp  = ScalarField(mesh)



    initialise!(rho, solid.rho.values)
    initialise!(k, solid.k.values)
    initialise!(kf, solid.k.values)
    initialise!(rDf, 1.0/solid.k.values)
    initialise!(rhocp, solid.rho.values .* solid.cp.values)

    LaplaceEnergy(T, Tf, rDf, rho, k, kf, cp, rhocp)
end


function initialise(
    energy::LaplaceEnergy, model::Physics{T1,F,S,M,Tu,E,D,BI}, _config
) where {T1,F,S,M,Tu,E,D,BI}

    # k = model.solid.k
    return nothing
end




function energy!(
    energy::LaplaceEnergy, model::Physics{T1,F,S,M,Tu,E,D,BI}, config
) where {T1,F,S,M,Tu,E,D,BI}
    return nothing
end