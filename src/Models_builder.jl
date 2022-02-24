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

# Model definitions
@model_builder "SteadyDiffusion" 1 1
export SteadyDiffusion
