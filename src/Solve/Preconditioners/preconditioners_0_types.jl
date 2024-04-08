
export PreconditionerType, Preconditioner
export Jacobi, NormDiagonal, LDL, ILU0
export DILU, DILUprecon
export CUDA_IC0, CUDA_ILU2

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

struct CUDA_IC0 <: PreconditionerType end
Adapt.@adapt_structure CUDA_IC0
struct CUDA_ILU2 <: PreconditionerType end
Adapt.@adapt_structure CUDA_ILU2

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


Preconditioner{CUDA_IC0}(A::CuSparseMatrixCSC{F,I}) where {F,I} = begin
    m, n = size(A)
    backend = get_backend(A)
    z = _convert_array!(zeros(F, n), backend)
    S = ic02(A)
    U1 = UpperTriangular(S)'
    U2 = UpperTriangular(S)

    function ldiv_ic0!(S::CuSparseMatrixCSC, x, y, z, U1, U2)
        ldiv!(z, U1, x)  # Forward substitution with L
        ldiv!(y, U2, z)   # Backward substitution with Lᴴ
        return y
    end

    P = LinearOperator(T, n, n, true, true, (y, x) -> ldiv_ic0!(S, x, y, z, U1, U2))

    Preconditioner{CUDA_IC0,typeof(A),typeof(P),typeof(S)}(A,P,S)
end

Preconditioner{CUDA_ILU2}(A::CuSparseMatrixCSC{F,I}) where {F,I} = begin
    m, n = size(A)
    backend = get_backend(A)
    z = _convert_array!(zeros(F, n), backend)
    S = ilu02(A)
    L1 = LowerTriangular(S)
    U1 = UnitUpperTriangular(S)

    function ldiv_ilu0!(S::CuSparseMatrixCSC, x, y, z, L1, U1)
        ldiv!(z, L1, x)      # Forward substitution with L
        ldiv!(y, U1, z)  # Backward substitution with U
        return y
      end

    P = LinearOperator(T, n, n, false, false, (y, x) -> ldiv_ilu0!(S, x, y, z, L1, U1))

    Preconditioner{CUDA_ILU2,typeof(A),typeof(P),typeof(S)}(A,P,S)
end