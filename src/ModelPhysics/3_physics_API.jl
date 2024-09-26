export change

"""
    change(model::Physics, property, value) => updatedModel::Physics

A convenience function to change properties of an exisitng `Physics` model.

# Input arguments
- `model::Physics` a `Physics` model to modify
- `property` is a symbol specifying the property to change 
- `value` is the new setting for the specified `property`

# Output

This function return a new `Physics` object

# Example

To change a model to run a transient simulation e.g. after converging in steady state

    modelTransient = change(model, :time, Transient())
"""
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