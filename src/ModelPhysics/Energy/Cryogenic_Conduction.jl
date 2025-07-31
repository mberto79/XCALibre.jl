export CryogenicConduction

"""
    CryogenicConduction <: AbstractEnergyModel

Type that represents energy conduction model for solids at cryogenic temperatures (works well between 4 and 300 K).

### Fields
- `T`       : Temperature scalar field on cells (ScalarField).
- `Tf`      : Temperature scalar field on faces (FaceScalarField).
- `rDf`     : Thermal diffusivity reciprocal on faces e.g. 1/k (FaceScalarField).
- `rhocp`   : Product of density and specific heat capacity on cells (ScalarField).
- `k`       : Thermal conductivity on cells (ScalarField).
- `kf`      : Thermal conductivity on faces (FaceScalarField).
- `cp`      : Specific heat capacity on cells (ScalarField).
- `material`: Symbol identifying the material type.
- `rho`     : Density.
"""
struct CryogenicConduction{S1,F1,F2,S2,S3,F3,S4,C1,C2} <: AbstractEnergyModel
    T::S1
    Tf::F1
    rDf::F2            
    rhocp::S2
    k::S3
    kf::F3
    cp::S4
    material::C1        
    rho::C2
end
Adapt.@adapt_structure CryogenicConduction


Energy{CryogenicConduction}(; material::Symbol, rho::Float64) = begin # maybe assign rho based on the material?
    args = (material, rho)
    ARGS = typeof(args)
    Energy{CryogenicConduction,ARGS}(args)
end

(energy::Energy{EnergyModel, ARG})(mesh, solid) where {EnergyModel<:CryogenicConduction,ARG} = begin
    material, rho = energy.args

    T  = ScalarField(mesh)
    Tf = FaceScalarField(mesh)

    rDf = FaceScalarField(mesh)
    rhocp  = ScalarField(mesh)

    k  = ScalarField(mesh)
    kf = FaceScalarField(mesh)

    cp  = ScalarField(mesh)

    CryogenicConduction(T, Tf, rDf, rhocp, k, kf, cp, material, rho)
end


# Perhaps need to pass this as a list of arguments
function initialise(
    energy::CryogenicConduction, model::Physics{T1,F,S,M,Tu,E,D,BI}, T_field, rDf, rhocp_field, k, kf, cp, rho, material, config
) where {T1,F,S,M,Tu,E,D,BI}

    # mesh = model.domain

    k_vals, cp_vals = get_coefficients(material, T_field)

    k.values .= k_vals
    cp.values .= cp_vals

    interpolate_harmonic!(kf, k, config) # harmonic interpolation instead of linear (better for k)

    initialise!(rhocp_field, rho)
    #initialise!(rhocp_field, rho .* cp) TRY THIS
    initialise!(rDf, 1.0)

    @. rhocp_field.values *= cp.values
    @. rDf.values *= (1.0/kf.values)

    return nothing
end




function energy!(
    energy::CryogenicConduction, model::Physics{T1,F,S,M,Tu,E,D,BI}, T_field, rDf, rhocp_field, k, kf, cp, rho, material, config
) where {T1,F,S,M,Tu,E,D,BI}

    # mesh = model.domain

    k_vals, cp_vals = get_coefficients(material, T_field)
    
    k.values .= k_vals
    cp.values .= cp_vals

    interpolate_harmonic!(kf, k, config) # harmonic interpolation instead of linear (better for k)

    initialise!(rhocp_field, rho)
    initialise!(rDf, 1.0)

    @. rhocp_field.values *= cp.values
    @. rDf.values *= (1.0/kf.values)

    return nothing
end