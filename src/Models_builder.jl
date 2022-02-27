export discretise!

macro build_model(modelName::String, terms::Integer, sources::Integer)
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

# Model definitions
@build_model "SteadyDiffusion" 1 1
export SteadyDiffusion
@discretise SteadyDiffusion 1 1
