
struct Discretisation{F1,F2,F3}
    ap!::F1
    an!::F2 
    b!::F3 
end

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

abstract type AbstractLaplacian end
abstract type AbstractScheme end
struct Laplacian{T<:AbstractScheme} <: AbstractLaplacian end
struct Linear <: AbstractScheme end

function Laplacian{Linear}(J, phi)
    ap!(i) = J*phi[i]
    an!(i) = J*phi[i]/2
    b!(i)  = 0.0
    Discretisation(ap!, an!, b!)
end

phi = [i for i ∈ 1:10000000]
U = [2 for _ ∈ 1:10000000]

term1 = Laplacian{Linear}(1, phi)
term2 = Laplacian{Linear}(2, phi)
@time model = term1 + term2

A = zeros(Float64, length(phi), 2)
A
function test!(A, phi, term)
    for i ∈ eachindex(phi)
        A[i, 1] += term.ap!(i)
        A[i, 2] += term.an!(i)
    end
end

@time test!(A, phi, model)