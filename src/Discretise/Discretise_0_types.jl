export AbstractField, AbstractScalarField, AbstractSource
export AbstractOperators, AbstractLaplacian, AbstractDivergence 
export AbstractScheme 
export AbstractLaplacian, AbstractDivergence 
export Laplacian, Divergence
export Constant, Linear, Upwind 
export ScalarField
export Equation 
# export Discretisation, Equation 

abstract type AbstractField end
abstract type AbstractScalarField <: AbstractField end
abstract type AbstractSource <: AbstractField end
abstract type AbstractOperator end
abstract type AbstractLaplacian <: AbstractOperator end
abstract type AbstractDivergence <: AbstractOperator end

# Supported discretisation schemes
abstract type AbstractScheme end
struct Constant <: AbstractScheme end
struct Linear <: AbstractScheme end
struct Upwind <: AbstractScheme end

# Fields
struct ScalarField{I,F} <: AbstractScalarField
    values::Vector{F}
    mesh::Mesh2{I,F}
end
ScalarField(mesh::Mesh2{I,F}) where {I,F} =begin
    ncells  = length(mesh.cells)
    ScalarField(zeros(F,ncells), mesh)
end

# Supported operators
struct Laplacian{T<:AbstractScheme} <: AbstractLaplacian
    J::Float64
    phi::ScalarField
    sign::Vector{Int64}
end

struct Divergence{T<:AbstractScheme} <: AbstractDivergence
    J::SVector{3, Float64}
    phi::ScalarField
    sign::Vector{Int64}
end

# Types 
# struct Discretisation{F,T,I,F1,F2,F3}
#     phi::F
#     terms::Vector{T}
#     signs::Vector{I}
#     ap!::F1
#     an!::F2 
#     b!::F3 
# end

struct Equation{Ti,Tf}
    A::SparseMatrixCSC{Tf,Ti}
    b::Vector{Tf}
    R::Vector{Tf}
    Fx::Vector{Tf}
end
Equation(mesh::Mesh2{Ti,Tf}) where {Ti,Tf} = begin
    nCells = length(mesh.cells)
    i, j, v = sparse_matrix_connectivity(mesh)
    Equation(sparse(i, j, v), zeros(Tf, nCells), zeros(Tf, nCells), zeros(Tf, nCells))
end

function sparse_matrix_connectivity(mesh::Mesh2{I,F}) where{I,F}
    cells = mesh.cells
    nCells = length(cells)
    i = I[]
    j = I[]
    for cID = 1:nCells   
        cell = cells[cID]
        push!(i, cID) # diagonal row index
        push!(j, cID) # diagonal column index
        for fi ∈ eachindex(cell.facesID)
            neighbour = cell.neighbours[fi]
            push!(i, cID) # cell index (row)
            push!(j, neighbour) # neighbour index (column)
        end
    end
    v = zeros(F, length(i))
    return i, j, v
end