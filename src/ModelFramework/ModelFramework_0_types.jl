export AbstractOperator, AbstractSource   
export Operator, Source, Src
export Time, Laplacian, Divergence, Si
export Model, Equation, ModelEquation
export nzval_index

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
# Source(f::T) where T = Src(f, 1, typeof(f))
Source(f::T) where T = Src(f, 1)
# Source(f::ScalarField) = Src(f.values, 1, typeof(f))
# Source(f::Number) = Src(f.values, 1, typeof(f)) # To implement!!

# MODEL TYPE
struct Model{TN,SN,T,S}
    # equation::E
    terms::T
    sources::S
end
function Adapt.adapt_structure(to, itp::Model{TN,SN}) where {TN,SN}
    terms = Adapt.adapt_structure(to, itp.terms); T = typeof(terms)
    sources = Adapt.adapt_structure(to, itp.sources); S = typeof(sources)
    Model{TN,SN,T,S}(terms, sources)
end
Model{TN,SN}(terms::T, sources::S) where {TN,SN,T,S} = begin
    Model{TN,SN,T,S}(terms, sources)
end
# Model(eqn::E, terms::T, sources::S, TN, SN) where {E,T,S} = begin
#     Model{E,T,S,TN,SN}(eqn, terms, sources)
# end

# Linear system matrix equation

## ORIGINAL STRUCTURE PARAMETERISED FOR GPU
struct Equation{VTf<:AbstractVector, ASA<:AbstractSparseArray}
    A::ASA
    b::VTf
    R::VTf
    Fx::VTf
    # mesh::Mesh2{Ti,Tf}
end
Adapt.@adapt_structure Equation
Equation(mesh::AbstractMesh) = begin
    nCells = length(mesh.cells)
    Tf = _get_float(mesh)
    mesh_temp = adapt(CPU(), mesh) # WARNING: Temp solution (sparse_matrix_connectivity should be kernel!!!!!!!)
    i, j, v = sparse_matrix_connectivity(mesh_temp)
    backend = _get_backend(mesh)
    Equation(
        _convert_array!(sparse(i, j, v), backend) ,
        _convert_array!(zeros(Tf, nCells), backend),
        _convert_array!(zeros(Tf, nCells), backend),
        _convert_array!(zeros(Tf, nCells), backend)
        # mesh
        )
end

## NEW STRUCTURE USED
# struct Equation{SMCSC,VTf}
#     A::SMCSC
#     b::VTf
#     R::VTf
#     Fx::VTf
#     # mesh::Mesh2{Ti,Tf}
# end
# Equation(mesh::Mesh2) = begin
#     nCells = length(mesh.cells)
#     Tf = _get_float(mesh)
#     backend = _get_backend(mesh)
#     i, j, v = sparse_matrix_connectivity(mesh)
#     A = _convert_array!(sparse(i, j, v), backend); SMCSC = typeof(A)
#     b = _convert_array!(zeros(Tf, nCells), backend); VTf = typeof(b)
#     R = _convert_array!(zeros(Tf, nCells), backend)  
#     Fx = _convert_array!(zeros(Tf, nCells), backend)
#     Equation{SMCSC,VTf}(
#         A,
#         b,
#         R,
#         Fx
#         # mesh
#         )
# end

function sparse_matrix_connectivity(mesh::AbstractMesh)
    (; cells, cell_neighbours) = mesh
    nCells = length(cells)
    TI = _get_int(mesh) # would this result in regression (type identified inside func?)
    TF = _get_float(mesh) # would this result in regression (type identified inside func?)
    i = TI[]
    j = TI[]
    for cID = 1:nCells   
        cell = cells[cID]
        push!(i, cID) # diagonal row index
        push!(j, cID) # diagonal column index
        # for fi ∈ eachindex(cell.facesID)
        for fi ∈ cell.faces_range
            # neighbour = cell.neighbours[fi]
            neighbour = cell_neighbours[fi]
            push!(i, cID) # cell index (row)
            push!(j, neighbour) # neighbour index (column)
        end
    end
    v = zeros(TF, length(i))
    return i, j, v
end

# function sparse_matrix_connectivity(mesh::Mesh2)
#     (; cells, cell_neighbours, cell_faces) = mesh
#     nCells = length(cells)
#     nFaces = length(cell_faces)
#     TI = _get_int(mesh) # would this result in regression (type identified inside func?)
#     TF = _get_float(mesh) # would this result in regression (type identified inside func?)
#     backend = _get_backend(mesh)

#     i = zeros(TI, nCells + nFaces)
#     i = adapt(backend, i)

#     j = zeros(TI, nCells + nFaces)
#     j = adapt(backend, j)

#     v = zeros(TF, nCells + nFaces)
#     v = adapt(backend, v)

#     kernel! = sparse_matrix_connectivity_kernel!(backend)
#     kernel!(i, j, cell_neighbours, cells, ndrange = nCells)
#     return i, j, v
# end

# @kernel function sparse_matrix_connectivity_kernel!(i_array, j_array, cell_neighbours, cells)
#     i = @index(Global)

#     faces_range_current = cells[i].faces_range
    
#     if i > 1
#         faces_range_prev = cells[i-1].faces_range
#     else
#         faces_range_prev = 0
#     end

#     cID = i + maximum(faces_range_prev)

#     for fi ∈ 1:length(faces_range_current)
#         neighbour = cell_neighbours[fi]
#         index = cID + fi

#         i_array[index] = cID
#         j_array[index] = neighbour
#     end
# end

function nzval_index(colptr, rowval, start_index, required_index, ione)
    start = colptr[start_index]
    offset = 0
    for j in start:length(rowval)
        offset += 1
        if rowval[j] == required_index
            break
        end
    end
    return start + offset - ione
end

# Model equation type 

struct ModelEquation{M,E,S,P}
    model::M 
    equation::E 
    solver::S
    preconditioner::P
end
Adapt.@adapt_structure ModelEquation