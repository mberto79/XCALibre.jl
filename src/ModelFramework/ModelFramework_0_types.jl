export AbstractOperator, AbstractSource   
export Operator, Source, Src
export Time, Laplacian, Divergence, Si
export Model, Equation, ModelEquation

# ABSTRACT TYPES 

abstract type AbstractSource end
abstract type AbstractOperator end

# OPERATORS

# Base Operator

struct Operator{F,P,S,T} <: AbstractOperator
    flux::F
    phi::P 
    sign::S
    type::T
end

# operators

struct Time{T} end
struct Laplacian{T}  end
struct Divergence{T} end
struct Si end

# constructors

Time{T}(flux, phi) where T = Operator(
    flux, phi, 1, Time{T}()
    )

Time{T}(phi) where T = Operator(
    ConstantScalar(one(_get_int(phi.mesh))), phi, 1, Time{T}()
    )

Laplacian{T}(flux, phi) where T = Operator(
    flux, phi, 1, Laplacian{T}()
    )

Divergence{T}(flux, phi) where T = Operator(
    flux, phi, 1, Divergence{T}()
    )

Si(flux, phi) = Operator(
    flux, phi, 1, Si()
)

# SOURCES

# Base Source
struct Src{F,S,T} <: AbstractSource
    field::F 
    sign::S 
    type::T
end

# Source types

struct Source end

Source(f::T) where T = Src(f, 1, typeof(f))
# Source(f::ScalarField) = Src(f.values, 1, typeof(f))
# Source(f::Number) = Src(f.values, 1, typeof(f)) # To implement!!

# MODEL TYPE
struct Model{T,S,TN,SN}
    # equation::E
    terms::T
    sources::S
end
Model{TN,SN}(terms::T, sources::S) where {T,S,TN,SN} = begin
    Model{T,S,TN,SN}(terms, sources)
end
# Model(eqn::E, terms::T, sources::S, TN, SN) where {E,T,S} = begin
#     Model{E,T,S,TN,SN}(eqn, terms, sources)
# end

# Linear system matrix equation

struct Equation{Ti,Tf}
    A::SparseMatrixCSC{Tf,Ti}
    b::Vector{Tf}
    R::Vector{Tf}
    Fx::Vector{Tf}
    # mesh::Mesh2{Ti,Tf}
end
Equation(mesh::Mesh2{Ti,Tf}) where {Ti,Tf} = begin
    nCells = length(mesh.cells)
    i, j, v = sparse_matrix_connectivity(mesh)
    Equation(
        sparse(i, j, v), 
        zeros(Tf, nCells), 
        zeros(Tf, nCells), 
        zeros(Tf, nCells)
        # mesh
        )
end

function sparse_matrix_connectivity(mesh::Mesh2)
    cells = mesh.cells
    nCells = length(cells)
    TI = _get_int(mesh) # would this result in regression (type identified inside func?)
    i = TI[]
    j = TI[]
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

# Model equation type 

struct ModelEquation{M,E,S,P}
    model::M 
    equation::E 
    solver::S
    preconditioner::P
end
