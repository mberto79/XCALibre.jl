export change

function change(model::Physics, property, value)
    @assert property ∈ fieldnames(Physics) throw(ArgumentError(
    """$value is not in Physics. 
    Use "fieldnames(Physics)" to find available properties""")
    )

    lens = opcompose(PropertyLens(property))
    updatedModel = set(model, lens, value)
    return updatedModel
end

function change(model::Physics, args...)
    updatedModel = nothing
    for arg in args
        updatedModel = change(model, arg...)
    end
    return updatedModel
end