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
term2 = Laplacian{Limited}(J1, phi, [1])
# term2 = VariableLaplacian{Limited}(J2, phi, [1])
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
            A[i] += terms.term1.J
            A[i] += terms.term2.J
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


macro model_builder_tuple(modelName::String, terms::Integer, sources::Integer)
    name = Symbol(modelName)

    param_types = []
    fieldsnames = []
    
    terms_tuple = Expr(:tuple)
    constructor_terms = Expr(:tuple)
    terms_types = Expr(:curly, :Tuple)
    for t ∈ 1:(terms)
        tt = Symbol("T$t") |> esc
        ts = Symbol("term$t")
        push!(param_types, tt)
        push!(terms_types.args, tt)
        push!(fieldsnames, ts)
        push!(constructor_terms.args, :($ts=$ts))
        push!(terms_tuple.args, QuoteNode(ts))
        # push!(fields, Expr(:(::), Symbol("term$t"), tt))
        
    end
    
    sources_tuple = Expr(:tuple)
    constructor_sources = Expr(:tuple)
    sources_types = Expr(:curly, :Tuple)
    for s ∈ 1:(sources)
        ss = Symbol("S$s") |> esc
        src_symbol = Symbol("source$s")
        push!(param_types, ss)
        push!(sources_types.args, ss)
        push!(fieldsnames, src_symbol)
        push!(constructor_sources.args, :($src_symbol=$src_symbol))
        push!(sources_tuple.args, QuoteNode(src_symbol))
        # push!(fields, Expr(:(::), Symbol("src$s"), ss))
    end

    terms = Expr(:(::), :terms, :(NamedTuple{$terms_tuple, $terms_types}))
    sources = Expr(:(::), :sources, :(NamedTuple{$sources_tuple, $sources_types}))

    structBody = Expr(:block, Expr(
        :struct, false, Expr(:curly, name, param_types...),
        Expr(:block, terms, sources)
        ))

    func = :(function $(name)($(fieldsnames...))
        begin
        $(name)(($(constructor_terms)),$(constructor_sources))
        end
    end) |> esc
    out = quote begin
        $structBody
        $func
    end end
    return  out #structBody, func
end

@macroexpand(@model_builder_tuple "SteadyDiffusion" 4 4)

@model_builder_tuple "SteadyDiffusion15" 4 4

model =SteadyDiffusion10((term1=term1,term2=term2), (source1=src1,))
model =SteadyDiffusion10(term1, term1, term1, src1)
model =SteadyDiffusion15(term1, term2, term1,  term2, src1, src1, src1, src1)
model.terms[2]
a = zeros(Int(100))

@time test_fn(a, model)
@code_warntype test_fn(a, model)
a

t = :((term1=term1, term2=t2,))

ti = Symbol("term1")

tt = :($ti=$ti)