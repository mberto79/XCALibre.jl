export get_phi, get_flux
export get_source, get_source_sign
export _A, _b

get_phi(eqn::ModelEquation)  = begin 
    eqn.model.terms[1].phi
end

get_phi(eqn::ModelEquation, ti::Integer)  = begin 
    eqn.model.terms[ti].phi
end

get_flux(eqn::ModelEquation, ti::Integer) = begin 
    eqn.model.terms[ti].flux
end

get_source(eqn::ModelEquation, ti::Integer) = begin 
    eqn.model.sources[ti].field
end

get_source_sign(eqn::ModelEquation, ti::Integer) = begin 
    eqn.model.sources[ti].sign
end

_A(eqn::ModelEquation) = eqn.equation.A
_b(eqn::ModelEquation) = eqn.equation.b