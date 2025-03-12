export _nut_access

struct MenterF1 #This is a dummy struct to allow the νₜ access function to be written ready for DES implementation
end

"""
    _nut_access(model::Physics{T,F,M,Tu,E,D,BI},turb) where {T,F,M,Tu<:AbstractTurbulenceModel,E,D,BI}

Access νₜ field from model.

### Input
- `model`  -- Physics model defined by user.
- `turb` -- Turbulence model to be used
###

### Output
- `nut`-- Eddy Viscosity ScalarField 
- `nutf`-- Eddy Viscosity FaceScalarField
###
"""
function _nut_access(model::Physics{T,F,M,Tu,E,D,BI},turb) where {T,F,M,Tu<:AbstractTurbulenceModel,E,D,BI}
    (;nut,nutf) = model.turbulence
    return (nut,nutf)
end

#Version for DES models
function _nut_access(model::Physics{T,F,M,Tu,E,D,BI},turb) where {T,F,M,Tu<:MenterF1,E,D,BI}
    if (turb::KOmegaModel) || (turb::KOmegaLKEModel) || (turb::LaminarModel)
        (;nut,nutf) = model.turbulence.rans
    elseif (turb::SmagorinskyModel)
        (;nut,nutf) = model.turbulence.les
    end
    return (nut,nutf)
end