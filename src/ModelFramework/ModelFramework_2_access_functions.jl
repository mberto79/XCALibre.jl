export get_phi, get_flux

get_phi(model::M) where M<:Model = model.terms[1].phi
get_flux(model::M, ti::Integer) where M<:Model = model.terms[ti].flux