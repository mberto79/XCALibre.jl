export get_phi, get_flux, get_source
export _A, _b

get_phi(model::Model)  = model.terms[1].phi
get_flux(model::Model, ti::Integer) = model.terms[ti].flux
get_source(model::Model, ti::Integer) = model.sources[ti].field
_A(model::M) where M = model.equation.A
_b(model::M) where M = model.equation.b