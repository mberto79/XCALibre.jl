export get_phi, get_flux
export get_source, get_source_sign
export _A, _b

get_phi(model::Model)  = model.terms[1].phi
get_flux(model::Model, ti::Integer) = model.terms[ti].flux
get_source(model::Model, ti::Integer) = model.sources[ti].field
get_source_sign(model::Model, ti::Integer) = model.sources[ti].sign
_A(model::M) where M = model.equation.A
_b(model::M) where M = model.equation.b