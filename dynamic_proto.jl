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
        :struct, false, Expr(:curly, name, parametricTypes...),
        Expr(:block, fields...)
        ))
    return structBody
end

@macroexpand(@model_builder "SteadyDiffusion" 2 1)

@model_builder "SteadyDiffusion" 2 1

struct Linear end
struct Limited end
struct Constant end

struct Laplacian{T}
    J::Float64
    phi::Vector{Float64}
    sign::Vector{Int64}
end

struct VariableLaplacian{T}
    J::Vector{Float64}
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
J2 = Float64[1.0,2.0,3.0,4.0]
sign = [1]

term1 = Laplacian{Linear}(J1, phi,[1])
term2 = VariableLaplacian{Limited}(J2, phi, [1])
src1 = Source{Constant}(phiSource, [1])

phiTerms = SteadyDiffusion(
    Laplacian{Linear}(J1, phi,[1]),
    VariableLaplacian{Limited}(J2, phi, [1]),
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

struct Model{T1,T2,S1}
    terms::NamedTuple{(:term1, :term2,), Tuple{T1,T2}}
    sources::NamedTuple{(:s1,), Tuple{S1}}
end

phiTuple = Model((term1=term1,term2=term2), (s1=src1,))

phiTuple.terms[2].J .= 2
phiModel.term2.J .= 2

phiTuple

function test_fn(A::Vector{Float64}, obj::SteadyDiffusion10)
    terms, sources = obj.terms, obj.sources
    # terms.term1
    # terms.term2
    # sources.source1
    for i ∈ 1:length(A)
        for ti ∈ 1:length(terms)
            A[i] += terms[ti].J
        end
    end
    # println(terms)
    # println(sources)
    nothing
end


function test_fn(obj::SteadyDiffusion)
    t1 = obj.term1
    t2 = obj.term2
    s1 = obj.source1
    
    nothing
end

@time test_fn(phiTuple)
@code_warntype test_fn(phiTuple)

@time test_fn(phiModel)
@code_warntype test_fn(phiModel)

t = Base.remove_linenums!( quote
    struct Test6{T1,T2,S1}
        terms::NamedTuple{(:term1, :term2,), Tuple{T1,T2}}
        sources::NamedTuple{(:s1,), Tuple{S1}}
    end
end)

ts = [:term1, :term2]
types = [:T2, :T2]
orig = :(NamedTuple{(:term1, :term2), Tuple{T1, T2}})
tup = Expr(:curly, :Tuple, types...)
tup0 = Expr(:curly, :Tuple)
push!(tup0.args, :T2)
ter0 = Expr(:tuple)
push!(ter0.args, QuoteNode(:HERE))
ter = Expr(:tuple, QuoteNode(:test0), QuoteNode(:test1))
interp = :(NamedTuple{$ter, $tup})

macro model_builder_tuple(modelName::String, terms::Integer, sources::Integer)
    name = Symbol(modelName)

    param_types = []
    
    terms_tuple = Expr(:tuple)
    terms_types = Expr(:curly, :Tuple)
    for t ∈ 1:(terms)
        tt = Symbol("T$t") |> esc
        push!(param_types, tt)
        push!(terms_types.args, tt)
        push!(terms_tuple.args, QuoteNode(Symbol("term$t")))
        # push!(fields, Expr(:(::), Symbol("term$t"), tt))
        
    end
    
    sources_tuple = Expr(:tuple)
    sources_types = Expr(:curly, :Tuple)
    for s ∈ 1:(sources)
        ss = Symbol("S$s") |> esc
        push!(param_types, ss)
        push!(sources_types.args, ss)
        push!(sources_tuple.args, QuoteNode(Symbol("source$s")))
        # push!(fields, Expr(:(::), Symbol("src$s"), ss))
    end

    terms = Expr(:(::), :terms, :(NamedTuple{$terms_tuple, $terms_types}))
    sources = Expr(:(::), :sources, :(NamedTuple{$sources_tuple, $sources_types}))

    structBody = Expr(:block, Expr(
        :struct, false, Expr(:curly, name, param_types...),
        Expr(:block, terms, sources)
        ))
    return structBody
end

@macroexpand(@model_builder_tuple "SteadyDiffusion10" 2 1)

@model_builder_tuple "SteadyDiffusion10" 2 1

model =SteadyDiffusion10((term1=term1,term2=term1), (source1=src1,))
model.terms[2]
a = zeros(Int(50e6))

@time test_fn(a, model)
@code_warntype test_fn(a, model)
a