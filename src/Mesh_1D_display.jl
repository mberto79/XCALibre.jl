
const T = Union{
    Vector{Node{F}},
    Vector{Element{I,F}},
    Vector{Cell{I,F}}
    } where {I <: Integer, F}

function plot(obj::T, propertySymbol::Symbol; labels=false) 
    N = length(obj)
    x = [getproperty(obj[i], propertySymbol)[1] for i ∈ 1:N]
    y = [getproperty(obj[i], propertySymbol)[2] for i ∈ 1:N]
    fig = scatter(x, y, aspect_ratio=:equal, label="Nodes")
    if labels
        for i ∈ 1:N
            annotate!(x[i], y[i], text("$i", 10))
        end
    end
    return fig
end

function plot!(fig, obj::T, propertySymbol::Symbol; labels=false) 
    N = length(obj)
    x = [getproperty(obj[i], propertySymbol)[1] for i ∈ 1:N]
    y = [getproperty(obj[i], propertySymbol)[2] for i ∈ 1:N]
    scatter!(fig, x, y, aspect_ratio=:equal, label="Nodes")
    if labels
        for i ∈ 1:N
            annotate!(fig, x[i], y[i], text("$i", 10))
        end
    end
    return fig
end