
Base.:(+)(a::Discretisation, b::Discretisation) = begin
    ap!(args...) = a.ap!(args...) + b.ap!(args...)
    an!(args...) = a.an!(args...) + b.an!(args...)
    b!(args...) = a.b!(args...) + b.b!(args...)
    terms = push!(a.terms, b.terms[1])
    signs = push!(a.signs, b.signs[1])
    Discretisation(a.phi, terms, signs, ap!, an!, b!)
end

Base.:(-)(a::Discretisation, b::Discretisation) = begin
    ap!(args...) = a.ap!(args...) - b.ap!(args...)
    an!(args...) = a.an!(args...) - b.an!(args...)
    b!(args...) = a.b!(args...) - b.b!(args...)
    terms = push!(a.terms, b.terms[1])
    signs = push!(a.signs, -b.signs[1])
    Discretisation(a.phi, terms, signs, ap!, an!, b!)
end

Base.:(==)(a::Discretisation, b::Real) = begin
    return a
end