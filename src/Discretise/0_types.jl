
export AbstractOperator, AbstractSource   
export Operator, Source, Src
export Laplacian, Divergence
export AbstractScheme, Constant, Linear, Upwind, Midpoint

export Model, Equation 
export AbstractBoundary, AbstractDirichlet, AbstractNeumann
export Dirichlet, Neumann 
export initialise!

# ABSTRACT TYPES 

abstract type AbstractSource end
abstract type AbstractOperator end

# SUPPORTED DISCRETISATION SCHEMES 

abstract type AbstractScheme end
struct Constant <: AbstractScheme end
struct Linear <: AbstractScheme end
struct Upwind <: AbstractScheme end
struct Midpoint <: AbstractScheme end

# OPERATORS

# Base Operator

struct Operator{F,P,S,T}
    flux::F
    phi::P 
    sign::S
    type::T
end

# operators

struct Laplacian{T} <: AbstractOperator end
struct Divergence{T} <: AbstractOperator end

# constructors

Laplacian{T}(flux, phi) where T = Operator(
    flux, phi, 1, Laplacian{T}()
    )

Divergence{T}(flux, phi) where T = Operator(
    flux, phi, 1, Divergence{T}()
    )

# SOURCES

# Base Source
struct Src{F,S,T}
    field::F 
    sign::S 
    type::T
end

# Source types

struct Source <: AbstractSource end

Source(f::AbstractVector) = Src(f, 1, typeof(f))
Source(f::ScalarField) = Src(f.values, 1, typeof(f))
# Source(f::Number) = Src(f.values, 1, typeof(f)) # To implement!!

# MODEL TYPE
struct Model{T,S, TN, SN}
    terms::T
    sources::S
end
Model{TN,SN}(terms::T, sources::S) where {T,S,TN,SN} = begin
    Model{T,S,TN,SN}(terms, sources)
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

struct Dirichlet{V}
    name::Symbol 
    value::V
    function Dirichlet(name, value::V) where {V}
        if V <: Number
            return new{eltype(value)}(name, value)
        elseif V <: Vector
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

struct Neumann{V}
    name::Symbol 
    value::V 
end