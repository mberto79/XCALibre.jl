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
export initialise!

# export Discretisation, Equation 

abstract type AbstractField end
abstract type AbstractScalarField <: AbstractField end
abstract type AbstractVectorField <: AbstractField end
abstract type AbstractSource <: AbstractField end
abstract type AbstractOperator end
abstract type AbstractLaplacian <: AbstractOperator end
abstract type AbstractDivergence <: AbstractOperator end

# SUPPORTED DISCRETISATION SCHEMES 

abstract type AbstractScheme end
struct Constant <: AbstractScheme end
struct Linear <: AbstractScheme end
struct Upwind <: AbstractScheme end

# FIELDS 

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

(s::AbstractScalarField)(i::Integer) = s.values[i]

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

(v::AbstractVectorField)(i::Integer) = SVector{3, eltype(v.x)}(v.x[i], v.y[i], v.z[i])

function initialise!(v::AbstractVectorField, vec::Vector{T}) where T
    n = length(vec)
    if T !== eltype(v.x)
        throw("Vectors are not the same type: $(eltype(v.x)) is not $T")
    elseif n == 3
        v.x .= vec[1]
        v.y .= vec[2]
        v.z .= vec[3]
    else
        throw("Vectors should have 3 components")
    end
    nothing
end

# OPERATORS

struct Laplacian{S<:AbstractScheme, T} <: AbstractLaplacian
    J::T # either Float64 or Vector{Float64}
    phi::ScalarField
    sign::Vector{Int64}
end

struct Divergence{S<:AbstractScheme, T} <: AbstractDivergence
    J::T # SVector{3, Float64} or Vector{SVector{3, Float64}}
    phi::ScalarField
    sign::Vector{Int64}
end

struct Equation{Ti,Tf}
    A::SparseMatrixCSC{Tf,Ti}
    b::Vector{Tf}
    R::Vector{Tf}
    Fx::Vector{Tf}
    mesh::Mesh2{Ti,Tf}
end
Equation(mesh::Mesh2{Ti,Tf}) where {Ti,Tf} = begin
    nCells = length(mesh.cells)
    i, j, v = sparse_matrix_connectivity(mesh)
    Equation(
        sparse(i, j, v), 
        zeros(Tf, nCells), 
        zeros(Tf, nCells), 
        zeros(Tf, nCells), 
        mesh
        )
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

# SUPPORTED BOUNDARY CONDITIONS 

abstract type AbstractBoundary end
abstract type AbstractDirichlet <: AbstractBoundary end
abstract type AbstractNeumann <: AbstractBoundary end

struct Dirichlet{F}
    name::Symbol 
    value::F 
    function Dirichlet(name, value::T) where {T}
        if T <: Number
            return new{eltype(value)}(name, value)
        elseif T <: Vector
            if length(value) == 3 
                nvalue = SVector{3, eltype(value)}(value)
                return new{typeof(nvalue)}(name, nvalue)
            else
                throw("Only vectors with three components can be used")
            end
        else
            throw("The value provided should be a scalar or a vector")
        end
    end
end

struct Neumann{F}
    name::Symbol 
    value::F 
end