export AbstractOperator, AbstractSource, AbstractEquation   
export Operator, Source
export Equation
export Time, Laplacian, Divergence, Si
export Model, ScalarMatrix, VectorMatrix, ScalarModel, VectorModel
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

struct Time{S,F,P}
    scheme::S
    flux::F 
    phi::P 
end
Adapt.@adapt_structure Time
Time{S}(phi) where S = Time(S(), ConstantScalar(one(_get_int(phi.mesh))), phi)
Time{S}(flux, phi) where S = Time(S(), flux, phi)

struct Laplacian{S,F,P}
    scheme::S
    flux::F 
    phi::P  
end
Adapt.@adapt_structure Laplacian 
Laplacian{S}(flux, phi) where S = Laplacian(S(), flux, phi)

struct Divergence{S,F,P}
    scheme::S
    flux::F 
    phi::P 
end
Adapt.@adapt_structure Divergence
Divergence{S}(flux, phi) where S = Divergence(S(), flux, phi)


struct Si{S,F,P}
    scheme::S
    flux::F 
    phi::P 
end
Adapt.@adapt_structure Si
Si(flux, phi) = Si(nothing, flux, phi)

# struct Time{T} end
# function Adapt.adapt_structure(to, itp::Time{T}) where {T}
#     Time{T}()
# end

# struct Laplacian{T} end
# function Adapt.adapt_structure(to, itp::Laplacian{T}) where {T}
#     Laplacian{T}()
# end

# struct Divergence{T} end
# function Adapt.adapt_structure(to, itp::Divergence{T}) where {T}
#     Divergence{T}()
# end

# struct Si end
# function Adapt.adapt_structure(to, itp::Si)
#     Si()
# end

# constructors

# Time{T}(flux, phi) where T = Operator(
#     flux, phi, 1, Time{T}()
#     )

# Time{T}(phi) where T = Operator(
#     ConstantScalar(one(_get_int(phi.mesh))), phi, 1, Time{T}()
#     )

# Laplacian{T}(flux, phi) where T = Operator(
#     flux, phi, 1, Laplacian{T}()
#     )

# Divergence{T}(flux, phi) where T = Operator(
#     flux, phi, 1, Divergence{T}()
#     )

# Si(flux, phi) = Operator(
#     flux, phi, 1, Si()
# )

# SOURCES

# Base Source
struct Source{F} <: AbstractSource
    field::F 
end
Adapt.@adapt_structure Source

# Source types

# struct Source end
# Adapt.@adapt_structure Source
# Source(f::T) where T = Src(f, 1)

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
struct ScalarMatrix{VTf<:AbstractVector, ASA<:AbstractSparseArray, OP} <: AbstractEquation
    A::ASA
    opA::OP
    b::VTf
    R::VTf
    Fx::VTf
end
Adapt.@adapt_structure ScalarMatrix

# Catch all function for fields that do not extend matrix
extend_matrix(mesh, BCs, i, j) = begin
    # mesh = field.mesh
    for BC ∈ BCs
        i, j = _extend_matrix(BC, mesh, i, j) # implemented in module Discretise for each BC
    end
    return i, j
end

# Catch all method for all non-constraint boundary conditions i.e. do not modify matrix
# Constraint-type BC that extend the sparse matrix should extend this method
_extend_matrix(BC, mesh, i, j) = begin
    return i, j
end

# ScalarMatrix(mesh::AbstractMesh) = begin
ScalarMatrix(phi::ScalarField, BCs) = begin
    mesh = phi.mesh
    nCells = length(mesh.cells)
    Tf = _get_float(mesh)
    mesh_temp = adapt(CPU(), mesh) # WARNING: Temp solution 
    i, j, v = sparse_matrix_connectivity(mesh_temp) # This needs to be a kernel
    i, j = extend_matrix(mesh, BCs, i, j)
    # i = [i; periodicConnectivity.i]
    # j = [j; periodicConnectivity.j]
    v = zeros(Tf, length(j))
    backend = _get_backend(mesh)
    # A = _convert_array!(sparse(i, j, v), backend)
    A = _build_A(backend, i, j, v, nCells)
    ScalarMatrix(
        A,

       _build_opA(A),
        # KP.KrylovOperator(A), # small gain in performance
        # A,

        # _convert_array!(zeros(Tf, nCells), backend),
        # _convert_array!(zeros(Tf, nCells), backend),
        # _convert_array!(zeros(Tf, nCells), backend)

        KernelAbstractions.zeros(backend, Tf, nCells),
        KernelAbstractions.zeros(backend, Tf, nCells),
        KernelAbstractions.zeros(backend, Tf, nCells)
        )
end

struct VectorMatrix{VTf<:AbstractVector, ASA<:AbstractSparseArray, OP} <: AbstractEquation
    A0::ASA
    A::ASA
    opA::OP
    bx::VTf
    by::VTf
    bz::VTf
    R::VTf
    Fx::VTf
end
Adapt.@adapt_structure VectorMatrix

VectorMatrix(psi::VectorField, BCs) = begin
    mesh = psi.mesh
    nCells = length(mesh.cells)
    Tf = _get_float(mesh)
    mesh_temp = adapt(CPU(), mesh) # WARNING: Temp solution 
    i, j, v = sparse_matrix_connectivity(mesh_temp) # This needs to be a kernel
    i, j = extend_matrix(mesh, BCs, i, j)
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
    VectorMatrix(
        A0,
        A,

        _build_opA(A),
        # KP.KrylovOperator(A),
        # A,

        # _convert_array!(zeros(Tf, nCells), backend),
        # _convert_array!(zeros(Tf, nCells), backend),
        # _convert_array!(zeros(Tf, nCells), backend),
        # _convert_array!(zeros(Tf, nCells), backend),
        # _convert_array!(zeros(Tf, nCells), backend)

        KernelAbstractions.zeros(backend, Tf, nCells),
        KernelAbstractions.zeros(backend, Tf, nCells),
        KernelAbstractions.zeros(backend, Tf, nCells),
        KernelAbstractions.zeros(backend, Tf, nCells),
        KernelAbstractions.zeros(backend, Tf, nCells)
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

struct Equation{E,S,P,M}
    matrix::E
    solver::S
    preconditioner::P
    mesh::M
end
Adapt.@adapt_structure Equation

Equation(psi::VectorField, BCs, solverSettings) = begin
    matrix = VectorMatrix(psi, BCs) 
    solver = _workspace(solverSettings.solver, matrix.bx)
    preconditioner = set_preconditioner(solverSettings.preconditioner, matrix.A)
    mesh = psi.mesh
    return Equation(matrix, solver, preconditioner, mesh)
end
Equation(phi::ScalarField, BCs, solverSettings) = begin
    matrix = ScalarMatrix(phi, BCs) 
    solver = _workspace(solverSettings.solver, matrix.b)
    preconditioner = set_preconditioner(solverSettings.preconditioner, matrix.A)
    mesh = phi.mesh
    return Equation(matrix, solver, preconditioner, mesh)
end

# TEMP LOCATION

export AbstractLinearSolver
export Cg, Cgs, Bicgstab, Gmres

abstract type AbstractLinearSolver end

struct Cg <: AbstractLinearSolver end
struct Cgs <: AbstractLinearSolver end
struct Bicgstab <: AbstractLinearSolver end
struct Gmres <: AbstractLinearSolver end

# Krylov.jl workspace constructors
_workspace(::Cg, b) = CgWorkspace(KrylovConstructor(b))
_workspace(::Cgs, b) = CgsWorkspace(KrylovConstructor(b))
_workspace(::Bicgstab, b) = BicgstabWorkspace(KrylovConstructor(b))
_workspace(::Gmres, b) = GmresWorkspace(KrylovConstructor(b))


export Preconditioner, PreconditionerType

abstract type PreconditionerType end

struct Preconditioner{T,M,P,S}
    A::M
    P::P
    storage::S
end
function Adapt.adapt_structure(to, itp::Preconditioner{T,M,Pr,S}) where {T,M,Pr,S}
    A = Adapt.adapt(to, itp.A)
    P = Adapt.adapt(to, itp.P)
    storage = Adapt.adapt(to, itp.storage) 
    Preconditioner{T,typeof(A),typeof(P),typeof(storage)}(A,P,storage)
end

set_preconditioner(PT::T, A) where T<:PreconditionerType = begin
    Preconditioner{T}(A)
end

# NEW TYPE TO HOLD DISCRETISATION FUNCTIONS 

struct LHS{S,SS,BC}
    scheme::S 
    scheme_source::SS
    apply_BCs::BC
end
Adapt.@adapt_structure LHS

struct RHS{SRC}
    source::SRC
end
Adapt.@adapt_structure RHS

struct Discretisation{S,SS,SRC,BC}
    scheme::S
    scheme_source::SS
    source::SRC
    apply_BCs::BC
end
Adapt.@adapt_structure Discretisation