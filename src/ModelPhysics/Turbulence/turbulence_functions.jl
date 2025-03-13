export _nut_access

"""
    _nut_access(model::Physics{T,F,M,Tu,E,D,BI},turb) where {T,F,M,Tu<:AbstractTurbulenceModel,E,D,BI}

Access νₜ field from model.

### Input
- `model`  -- Physics model defined by user.
- `turb` -- Turbulence model to be used
###

### Output
- `nut` -- Eddy Viscosity ScalarField 
- `nutf` -- Eddy Viscosity FaceScalarField
###
"""
function _nut_access(model::Physics{T,F,M,Tu,E,D,BI}, turb) where {T,F,M,Tu<:AbstractTurbulenceModel,E,D,BI}
    (; nut, nutf) = model.turbulence
    return (nut, nutf)
end

#Versions for DES models
function _nut_access(model::Physics{T,F,M,Tu,E,D,BI}, turb::Union{KOmegaModel,KOmegaLKEModel}) where {T,F,M,Tu<:MenterF1,E,D,BI}
    (; nut, nutf) = model.turbulence.rans
    return (nut, nutf)
end

function _nut_access(model::Physics{T,F,M,Tu,E,D,BI}, turb::Smagorinsky) where {T,F,M,Tu<:MenterF1,E,D,BI}
    (; nut, nutf) = model.turbulence.les
    return (nut, nutf)
end