export AbstractOperator, AbstractSource, AbstractEquation   
export Operator, Source, Src
export Time, Laplacian, Divergence, Si
export Model, ScalarEquation, VectorEquation, ModelEquation, ScalarModel, VectorModel
export nzval_index
export spindex, spindex_csc

# ABSTRACT TYPES 

abstract type AbstractSource end
abstract type AbstractOperator end
abstract type AbstractEquation end

# OPERATORS

# Base Operator

struct Operator{F,P,S,T} <: AbstractOperator
    flux::F
    phi::P 
    sign::S
    type::T
end
Adapt.@adapt_structure Operator

# operators

struct Time{T} end
function Adapt.adapt_structure(to, itp::Time{T}) where {T}
    Time{T}()
end

struct Laplacian{T} end
function Adapt.adapt_structure(to, itp::Laplacian{T}) where {T}
    Laplacian{T}()
end

struct Divergence{T} end
function Adapt.adapt_structure(to, itp::Divergence{T}) where {T}
    Divergence{T}()
end

struct Si end
function Adapt.adapt_structure(to, itp::Si)
    Si()
end

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
struct Src{F,S} <: AbstractSource
    field::F 
    sign::S 
    # type::T
end
Adapt.@adapt_structure Src

# Source types

struct Source end
Adapt.@adapt_structure Source
Source(f::T) where T = Src(f, 1)

# MODEL TYPE
struct Model{TN,SN,T,S}
    terms::T
    sources::S
end
# Adapt.@adapt_structure Model
function Adapt.adapt_structure(to, itp::Model{TN,SN,TT,SS}) where {TN,SN,TT,SS}
    terms = Adapt.adapt(to, itp.terms); T = typeof(terms)
    sources = Adapt.adapt(to, itp.sources); S = typeof(sources)
    Model{TN,SN,T,S}(terms, sources)
end
Model{TN,SN}(terms::T, sources::S) where {TN,SN,T,S} = begin
    Model{TN,SN,T,S}(terms, sources)
end

# Linear system matrix equation

# _build_A(backend::CPU, i, j, v, n) = sparsecsr(i, j, v, n, n)
# _build_opA(A::SparseMatricesCSR.SparseMatrixCSR) = LinearOperator(A)

_build_A(backend::CPU, i, j, v, n) = SparseXCSR(sparsecsr(i, j, v, n, n))
_build_opA(A::SparseXCSR) = A

## ORIGINAL STRUCTURE PARAMETERISED FOR GPU
struct ScalarEquation{VTf<:AbstractVector, ASA<:AbstractSparseArray, OP} <: AbstractEquation
    A::ASA
    opA::OP
    b::VTf
    R::VTf
    Fx::VTf
end
Adapt.@adapt_structure ScalarEquation

# Catch all function for fields that do not extend matrix
extend_matrix(field, i, j) = begin
    mesh = field.mesh
    for BC ∈ field.BCs
        i, j = _extend_matrix(BC, mesh, i, j) # implemented in module Discretise for each BC
    end
    return i, j
end

# Catch all method for all non-constraint boundary conditions i.e. do not modify matrix
# Constraint-type BC that extend the sparse matrix should extend this method
_extend_matrix(BC, mesh, i, j) = begin
    return i, j
end

# ScalarEquation(mesh::AbstractMesh) = begin
ScalarEquation(phi::ScalarField) = begin
    mesh = phi.mesh
    nCells = length(mesh.cells)
    Tf = _get_float(mesh)
    mesh_temp = adapt(CPU(), mesh) # WARNING: Temp solution 
    i, j, v = sparse_matrix_connectivity(mesh_temp) # This needs to be a kernel
    i, j = extend_matrix(phi, i, j)
    # i = [i; periodicConnectivity.i]
    # j = [j; periodicConnectivity.j]
    v = zeros(Tf, length(j))
    backend = _get_backend(mesh)
    # A = _convert_array!(sparse(i, j, v), backend)
    A = _build_A(backend, i, j, v, nCells)
    ScalarEquation(
        A,

       _build_opA(A),
        # KP.KrylovOperator(A), # small gain in performance
        # A,

        _convert_array!(zeros(Tf, nCells), backend),
        _convert_array!(zeros(Tf, nCells), backend),
        _convert_array!(zeros(Tf, nCells), backend)
        )
end

struct VectorEquation{VTf<:AbstractVector, ASA<:AbstractSparseArray, OP} <: AbstractEquation
    A0::ASA
    A::ASA
    opA::OP
    bx::VTf
    by::VTf
    bz::VTf
    R::VTf
    Fx::VTf
end
Adapt.@adapt_structure VectorEquation

VectorEquation(psi::VectorField) = begin
    mesh = psi.mesh
    nCells = length(mesh.cells)
    Tf = _get_float(mesh)
    mesh_temp = adapt(CPU(), mesh) # WARNING: Temp solution 
    i, j, v = sparse_matrix_connectivity(mesh_temp) # This needs to be a kernel
    i, j = extend_matrix(psi, i, j)
    # i = [i; periodicConnectivity.i]
    # j = [j; periodicConnectivity.j]
    v = zeros(Tf, length(j))
    backend = _get_backend(mesh)
    # A = _convert_array!(sparse(i, j, v), backend) 
    # A0 = _convert_array!(sparse(i, j, v), backend)
    # A = _convert_array!(sparsecsr(i, j, v), backend) 
    # A0 = _convert_array!(sparsecsr(i, j, v), backend)

    A = _build_A(backend, i, j, v, nCells)
    A0 = _build_A(backend, i, j, v, nCells)
    VectorEquation(
        A0,
        A,

        _build_opA(A),
        # KP.KrylovOperator(A),
        # A,

        _convert_array!(zeros(Tf, nCells), backend),
        _convert_array!(zeros(Tf, nCells), backend),
        _convert_array!(zeros(Tf, nCells), backend),
        _convert_array!(zeros(Tf, nCells), backend),
        _convert_array!(zeros(Tf, nCells), backend)
        )
end

Base.show(io::IO, model_eqn::AbstractEquation) = begin
    output = "Equation storage ready!"
    print(io, output)
end

# Sparse matrix connectivity function definition
function sparse_matrix_connectivity(mesh::AbstractMesh)
    (; cells, cell_neighbours) = mesh
    nCells = length(cells)
    TI = _get_int(mesh)
    TF = _get_float(mesh)
    i = TI[]
    j = TI[]
    for cID = 1:nCells   
        cell = cells[cID]
        push!(i, cID) # diagonal row index
        push!(j, cID) # diagonal column index
        for fi ∈ cell.faces_range
            neighbour = cell_neighbours[fi]
            push!(i, cID) # cell index (row)
            push!(j, neighbour) # neighbour index (column)
        end
    end
    v = zeros(TF, length(i))
    return i, j, v
end


# Sparse CSR format
function spindex(rowptr::AbstractArray{T}, colval, i, j) where T
    start_ind = rowptr[i]
    end_ind = rowptr[i+1] - one(T)

    ind = zero(T)
    for nzi in start_ind:end_ind
        if colval[nzi] == j
            ind = nzi
            break
        end
    end
    return ind
end

# Sparse CSC format
function spindex_csc(colptr::AbstractArray{T}, rowval, i, j) where T
    start_ind = colptr[j]
    end_ind = colptr[j+1] - one(T)

    ind = zero(T)
    for nzi in start_ind:end_ind
        if rowval[nzi] == i
            ind = nzi
            break
        end
    end
    return ind
end

# Model equation type 
struct ScalarModel end
Adapt.@adapt_structure ScalarModel

struct VectorModel end
Adapt.@adapt_structure VectorModel

struct ModelEquation{T,M,E,S,P}
    type::T
    model::M 
    equation::E 
    solver::S
    preconditioner::P
end
Adapt.@adapt_structure ModelEquation