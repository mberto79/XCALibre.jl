export get_phi, get_flux
export _A, _b

get_phi(model::M) where M<:Model = model.terms[1].phi
get_flux(model::M, ti::Integer) where M<:Model = model.terms[ti].flux
_A(model::M) where M = model.equation.A
_b(model::M) where M = model.equation.b