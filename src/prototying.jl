struct Upwind end
struct LinearUpwind end

struct Term0{T}
    a::Int64
    b::Float64
    type::T
end

struct Term1{T}
    a::Int64
    b::Float64
    c::Float64
    type::T
end
t0 = Term0{Upwind}(1,1.0,Upwind())
t1 = Term1{LinearUpwind}(1, 1.0, 2.0,LinearUpwind())

terms = [t0, t1]

Tterms = (terms...,)

isbits(Tterms)

typeof(Tterms)

struct TestIsBits
    terms
end

struct TestIsBits1
    terms::Tuple{FVM_1D.Term0{FVM_1D.Upwind}, FVM_1D.Term1{FVM_1D.LinearUpwind}}
end

struct TestIsBits2{T1,T2}
    terms::Tuple{T1, T2}
end

tt = TestIsBits2(Tterms)

isbits(tt)

macro terms(ex)
    terms = esc(ex)
    quote
        nTerms = length($terms)
        types = [Symbol("T$i") for i ∈ 1:nTerms]
        # T = :({$(types...)})
        braces = Expr(:braces, types...)
        structDef = Expr(:curly, :Ter, types...)
        typeDef = Expr(:curly, :Tuple, types...)

        t = quote
            struct $structDef
                terms::$typeDef
            end
        end 
        eval(t)
    end
end

ts = @terms terms

[Symbol("T$t") for t ∈ 1:5]
Expr(:curly, :term, [i for i ∈ [[Symbol("T$t") for t ∈ 1:5]...]]...)
i = 2
ttt = quote {T1,T2} end
dump(ttt)

struct Lin end
struct Operation0{T}
    Γ
    ϕ
    discretisation::T
    operator::Symbol
end
Operation0{Linear}(Γ, ϕ) = Operation0(Γ, ϕ, Lin(), :laplacian)

Op = Operation0{Linear}(1,2)

function build_term(op::Operation)
    op.operator