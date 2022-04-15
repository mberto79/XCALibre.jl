export AbstractField, AbstractScalarField, AbstractSource
export AbstractOperators, AbstractLaplacian, AbstractDivergence 
export AbstractScheme 
export AbstractLaplacian, AbstractDivergence 
export Laplacian, Divergence
export Constant, Linear, Upwind 
export ScalarField, FaceScalarField
export VectorField, FaceVectorField
export Equation 
export AbstractBoundary, AbstractDirichlet, AbstractNeumann
export Dirichlet, Neumann 
# export Discretisation, Equation 

abstract type AbstractField end
abstract type AbstractScalarField <: AbstractField end
abstract type AbstractVectorField <: AbstractField end
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

struct FaceScalarField{I,F} <: AbstractScalarField
    values::Vector{F}
    mesh::Mesh2{I,F}
end
FaceScalarField(mesh::Mesh2{I,F}) where {I,F} =begin
    nfaces  = length(mesh.faces)
    FaceScalarField(zeros(F,nfaces), mesh)
end

struct VectorField{I,F} <: AbstractVectorField
    x::Vector{F}
    y::Vector{F}
    z::Vector{F}
    mesh::Mesh2{I,F}
end
VectorField(mesh::Mesh2{I,F}) where {I,F} = begin
    ncells = length(mesh.cells)
    VectorField(zeros(F, ncells), zeros(F, ncells), zeros(F, ncells), mesh)
end

struct FaceVectorField{I,F} <: AbstractVectorField
    x::Vector{F}
    y::Vector{F}
    z::Vector{F}
    mesh::Mesh2{I,F}
end
FaceVectorField(mesh::Mesh2{I,F}) where {I,F} = begin
    nfaces = length(mesh.faces)
    FaceVectorField(zeros(F, nfaces), zeros(F, nfaces), zeros(F, nfaces), mesh)
end

(v::AbstractVectorField)(i::Integer) = SVector{3, typeof(v.x[1])}(v.x[i], v.y[i], v.z[i])

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

# Supported boundary conditions 
abstract type AbstractBoundary end
abstract type AbstractDirichlet <: AbstractBoundary end
abstract type AbstractNeumann <: AbstractBoundary end

struct Dirichlet{F}
    name::Symbol 
    value::F 
end

struct Neumann{F}
    name::Symbol 
    value::F 
end