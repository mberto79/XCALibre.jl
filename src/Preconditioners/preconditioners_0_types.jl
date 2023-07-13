
export PreconditionerType, Preconditioner
export Jacobi, NormDiagonal, LDL, ILU0

abstract type PreconditionerType end
struct NormDiagonal <: PreconditionerType end
struct Jacobi <: PreconditionerType end
struct LDL <: PreconditionerType end
struct ILU0 <: PreconditionerType end

struct Preconditioner{T,M,P,S}
    A::M
    P::P
    storage::S
end

Preconditioner{NormDiagonal}(A::SparseMatrixCSC{F,I}) where {F,I} = begin
    m, n = size(A)
    m == n || throw("Matrix not square")
    S = zeros(F, m)
    P = opDiagonal(S)
    Preconditioner{NormDiagonal,typeof(A),typeof(P),typeof(S)}(A,P,S)
end

Preconditioner{Jacobi}(A::SparseMatrixCSC{F,I}) where {F,I} = begin
    m, n = size(A)
    m == n || throw("Matrix not square")
    S = zeros(F, m)
    P = opDiagonal(S)
    Preconditioner{Jacobi,typeof(A),typeof(P),typeof(S)}(A,P,S)
end

Preconditioner{LDL}(A::SparseMatrixCSC{F,I}) where {F,I} = begin
    m, n = size(A)
    m == n || throw("Matrix not square")
    S = zeros(F, m)
    P  = opLDL(A)
    Preconditioner{LDL,typeof(A),typeof(P),typeof(S)}(A,P,S)
end

Preconditioner{ILU0}(A::SparseMatrixCSC{F,I}) where {F,I} = begin
    m, n = size(A)
    m == n || throw("Matrix not square")
    S = ilu0(A)
    P  = LinearOperator(
        Float64, m, n, false, false, (y, v) -> ldiv!(y, S, v)
        )
    Preconditioner{ILU0,typeof(A),typeof(P),typeof(S)}(A,P,S)
end
