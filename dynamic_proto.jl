struct Model0{T1,T2,S1}
    term1::T1 
    term2::T2 
    source1::S1
end

template = Base.remove_linenums!(
    quote 
        struct MODEL{T1,T2,S1}
            a::T1
            b::T2
            c::S1
        end
    end)

macro model_builder(modelName::String, terms::Integer, sources::Integer)
    name = Symbol(modelName)
    NT = Symbol(:NT) |> esc
    NS = Symbol(:NS) |> esc


    parametricTypes = []
    fields = []
    
    for t ∈ 1:(terms)
        tt = Symbol("T$t") |> esc
        push!(parametricTypes, tt)
        push!(fields, Expr(:(::), Symbol("term$t"), tt))
        
    end
    
    for s ∈ 1:(sources)
        ss = Symbol("S$s") |> esc
        push!(parametricTypes, ss)
        push!(fields, Expr(:(::), Symbol("src$s"), ss))
    end

    structBody = Expr(:block, Expr(
        :struct, false, Expr(:curly, name, NT, NS, parametricTypes...),
        Expr(:block, fields...)
        ))
    return structBody
end

@macroexpand(@model_builder "SteadyDiffusion" 2 1)

@model_builder "SteadyDiffusion1" 2 1

struct Linear end
struct Limited end
struct Constant end

struct Laplacian{T}
    J::Float64
    phi::Vector{Float64}
    sign::Vector{Int64}
end

struct Source{T}
    phi::Vector{Float64}
    sign::Vector{Int64}
end

cells = Int(10)
phi = ones(cells)
phiSource =  zeros(cells)
J1 = 0.5
J2 = 2.0
sign = [1]

term1 = Laplacian{Linear}(J1, phi,[1])
term2 = Laplacian{Limited}(J2, phi, [1])
source1 = Source{Constant}(phiSource, [1])

phiModel = SteadyDiffusion1{2,1}(
    Laplacian{Linear}(J1, phi,[1]),
    Laplacian{Limited}(J2, phi, [1]),
    Source{Constant}(phiSource, [1])
)

macro discretise_macro(ex)
    model = esc(ex)
    name = esc(:discretise_fn)
    A = esc(:A)

    insert = Expr(:block)
    for term ∈ [:(model.term1), :(model.term2)]
        push!(insert.args, :(A[i] += $(term).J) )
    end

    forLoop = :(for i ∈ eachindex(A); $insert; end)
    quote
    return function $name($A, $model::Model0)
        $forLoop
    end
    end
end

t = @macroexpand(@discretise_macro model)
discretise_macro! = @discretise_macro model
@time discretise_macro!(A, model)
A
A = Float64[1:cells;]
A

struct Test0{NT, NS, T1,S1}
    term::T1
    source::S1
end

struct Test7{T1,T2,S1}
    terms::NamedTuple{(:term1, :term2), Tuple{T1,T2}}
    sources::NamedTuple{(:source1), Tuple{S1}}
end

obj = Test7((term1=term1,term2=term2), (source1=source1))

function test_fn(obj.terms, obj.sources)
    nothing
end

