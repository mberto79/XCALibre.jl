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
    I::Vector{Ti}
    J::Vector{Ti}
    vals::Vector{Tf}
end
Equation(mesh::Mesh2{Ti,Tf}) where {Ti,Tf} = begin
    nCells = length(mesh.cells)
    i, j, v = sparse_matrix_connectivity(mesh)
    Equation(sparse(i, j, v), zeros(Tf, nCells), i, j, zeros(Tf, length(v)))
end

function sparse_matrix_connectivity(mesh::Mesh2{I,F}) where{I,F}
    cells = mesh.cells
    faces = mesh.faces
    nCells = length(cells)
    i = I[]
    j = I[]
    for cID = 1:nCells   
        cell = cells[cID]
        push!(i, cID)
        push!(j, cID)
        for fi ∈ eachindex(cell.facesID)
            fID = cell.facesID[fi]
            face = faces[fID]
            neighbour = cell.neighbours[fi]
            # c1 = face.ownerCells[1]
            # c2 = face.ownerCells[2]
            # push!(i, cID)
            # push!(j, cID)
            # A[cID, cID] += ...
            # if c1 != c2
                # A[cID, neighbour] += ...
                push!(i, cID)
                push!(j, neighbour)
            # end
        end
    end
    v = zeros(F, length(i))
    return i, j, v
end