
Base.:(+)(a::Discretisation, b::Discretisation) = begin
    ap!(i) = a.ap!(i) + b.ap!(i)
    an!(i) = a.an!(i) + b.an!(i)
    b!(i) = a.b!(i) + b.b!(i)
    Discretisation(ap!, an!, b!)
end

Base.:(-)(a::Discretisation, b::Discretisation) = begin
    ap!(i) = a.ap!(i) - b.ap!(i)
    an!(i) = a.an!(i) - b.an!(i)
    b!(i) = a.b!(i) - b.b!(i)
    Discretisation(ap!, an!, b!)
end