
_add(::T, a, b) where T = Operator(
    a.phi, (a.flux, b.flux), (1, 1), (a.type, b.type)
    )
_add(::T, a, b) where T<: Tuple = Operator(
    a.phi, (a.flux..., b.flux), (a.sign..., 1), (a.type..., b.type)
    )
_substract(::T, a, b) where T = Operator(
    a.phi, (a.flux, b.flux), (1, -1), (a.type, b.type)
    )
_substract(::T, a, b) where T<: Tuple = Operator(
    a.phi, (a.flux..., b.flux), (a.sign..., -1), (a.type..., b.type)
    )

Base.:+(a::Operator, b::Operator) = _add(a.type, a, b)
Base.:-(a::Operator, b::Operator) = _substract(a.type, a, b)

Base.:(==)(a::Operator, b::Src) = Model(a,b)