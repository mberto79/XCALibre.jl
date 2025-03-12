export _nut_access

struct MenterF1
end

function _nut_access(model::Physics{T,F,M,Tu,E,D,BI},turb) where {T,F,M,Tu<:AbstractTurbulenceModel,E,D,BI}
    println("turbulence model is of type: ", typeof(turb))
    (;nut,nutf) = model.turbulence
    return (nut,nutf)
end

function _nut_access(model::Physics{T,F,M,Tu,E,D,BI},turb) where {T,F,M,Tu<:MenterF1,E,D,BI}
    

end