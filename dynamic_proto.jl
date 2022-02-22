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

@model_builder "SteadyDiffusion2" 2 1
SteadyDiffusion2(1,2,3)

struct Linear end
struct Limited end
struct Constant end

struct Laplacian{T}
    J::Float64
    phi::Vector{Float64}
    sign::Vector{Int64}
end

struct Source1{T}
    phi::Vector{Float64}
    sign::Vector{Int64}
end

cells = Int(50e6)
phi = ones(cells)
phiSource =  zeros(cells)
J1 = 0.5
J2 = 2.0
sign = [1]

model = Model0(
    Laplacian{Linear}(J1, phi,[1]),
    Laplacian{Limited}(J2, phi, [1]),
    Source1{Constant}(phiSource, [1])
)


function discretise!(A, model::Model0)
    terms = (model.term1, model.term2)
    for term in terms
        for i ∈ eachindex(A)
            A[i] += term.J
        end
    end
end

terms = (Laplacian{Linear}(J1, phi,[1]), Laplacian{Limited}(J2, phi, [1]))
function discretise_tuple!(A, terms)
    for term in terms
        for i ∈ eachindex(A)
            A[i] += term.J
        end
    end
end

function discretise_fused!(A, model::Model0)
    # terms = (model.term1, model.term2)
    # for term in terms
    for i ∈ eachindex(A)
        A[i] += model.term1.J
        A[i] += model.term2.J
    end
    # end
end

GC.gc()
A = Float64[1:cells;]
@time discretise!(A,model)
A = Float64[1:cells;]
@time discretise_tuple!(A, terms)
A = Float64[1:cells;]
@time discretise_fused!(A,model)
A = Float64[1:cells;]
@time discretise_macro!(A, model)
A = Float64[1:cells;]
A = nothing

A

terms = (Laplacian{Linear}(J1, phi,[1]), Laplacian{Limited}(J2, phi, [1]))
typeof(terms)

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