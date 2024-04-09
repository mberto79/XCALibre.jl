
export PreconditionerType, Preconditioner
export Jacobi, NormDiagonal, LDL, ILU0
export DILU, DILUprecon

abstract type PreconditionerType end
struct NormDiagonal <: PreconditionerType end
Adapt.@adapt_structure NormDiagonal
struct Jacobi <: PreconditionerType end
Adapt.@adapt_structure Jacobi
struct LDL <: PreconditionerType end
Adapt.@adapt_structure LDL
struct ILU0 <: PreconditionerType end
Adapt.@adapt_structure ILU0
struct DILU <: PreconditionerType end
Adapt.@adapt_structure DILU


struct Preconditioner{T,M,P,S}
    A::M
    P::P
    storage::S
end
function Adapt.adapt_structure(to, itp::Preconditioner{T}) where {T}
    A = Adapt.adapt_structure(to, itp.A)
    P = Adapt.adapt_structure(to, itp.P)
    storage = Adapt.adapt_structure(to, itp.storage) 
    Preconditioner{T,typeof(A),typeof(P),typeof(storage)}(A,P,storage)
end
Preconditioner{NormDiagonal}(A::AbstractSparseArray{F,I}) where {F,I} = begin
    backend = get_backend(A)
    m, n = size(A)
    m == n || throw("Matrix not square")
    S = _convert_array!(zeros(m), backend)
    P = opDiagonal(S)
    Preconditioner{NormDiagonal,typeof(A),typeof(P),typeof(S)}(A,P,S)
end

Preconditioner{Jacobi}(A::AbstractSparseArray{F,I}) where {F,I} = begin
    backend = get_backend(A)
    m, n = size(A)
    m == n || throw("Matrix not square")
    S = _convert_array!(zeros(m), backend)
    P = opDiagonal(S)
    Preconditioner{Jacobi,typeof(A),typeof(P),typeof(S)}(A,P,S)
end

Preconditioner{LDL}(A::AbstractSparseArray{F,I}) where {F,I} = begin
    m, n = size(A)
    m == n || throw("Matrix not square")
    S = zeros(F, m)
    P  = opLDL(A)
    Preconditioner{LDL,typeof(A),typeof(P),typeof(S)}(A,P,S)
end

Preconditioner{ILU0}(A::AbstractSparseArray{F,I}) where {F,I} = begin
    m, n = size(A)
    m == n || throw("Matrix not square")
    S = ilu0(A)
    P  = LinearOperator(
        F, m, n, false, false, (y, v) -> ldiv!(y, S, v)
        )
    Preconditioner{ILU0,typeof(A),typeof(P),typeof(S)}(A,P,S)
end

struct DILUprecon{M,V,VI,VUR}
    A::M
    D::V
    Di::VI
    Ri::VI
    J::VI
    upper_indices_IDs::VUR
end
Adapt.@adapt_structure DILUprecon
Preconditioner{DILU}(A::AbstractSparseArray{F,I}) where {F,I} = begin
    m, n = size(A)
    m == n || throw("Matrix not square")
    D = zeros(F, m)
    Di = zeros(I, m)
    diagonal_indices!(Di, A)
    @time Ri, J, upper_indices_IDs = upper_row_indices(A, Di)
    S = DILUprecon(A, D, Di, Ri, J, upper_indices_IDs)
    P  = LinearOperator(
        F, m, n, false, false, (y, v) -> ldiv!(y, S, v)
        )
    Preconditioner{DILU,typeof(A),typeof(P),typeof(S)}(A,P,S)
end
