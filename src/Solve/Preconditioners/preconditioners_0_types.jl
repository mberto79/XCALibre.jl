
export Preconditioner, PreconditionerType
export Jacobi, NormDiagonal #, ILU0 # , LDL
export DILU, DILUprecon
export IC0GPU, ILU0GPU

abstract type PreconditionerType end
abstract type LDIVPreconditioner <: PreconditionerType end
abstract type MULPreconditioner <: PreconditionerType end

struct NormDiagonal <: MULPreconditioner end
Adapt.@adapt_structure NormDiagonal

struct Jacobi <: MULPreconditioner end
Adapt.@adapt_structure Jacobi

# struct LDL <: MULPreconditioner end
# Adapt.@adapt_structure LDL

# struct ILU0 <: MULPreconditioner end
# Adapt.@adapt_structure ILU0

struct DILU <: LDIVPreconditioner end
Adapt.@adapt_structure DILU

struct IC0GPU <: LDIVPreconditioner end
Adapt.@adapt_structure IC0GPU

struct ILU0GPU <: LDIVPreconditioner end
Adapt.@adapt_structure ILU0GPU

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

is_ldiv(precon::Preconditioner{T,M,P,S}) where {T,M,P,S} = T <: LDIVPreconditioner

Preconditioner{NormDiagonal}(A::AbstractSparseArray{F,I}) where {F,I} = begin
    backend = get_backend(A)
    m, n = size(A)
    m == n || throw("Matrix not square")
    S = _convert_array!(zeros(m), backend)
    P = opDiagonal(S)
    Preconditioner{NormDiagonal,typeof(A),typeof(P),typeof(S)}(A,P,S)
end

Preconditioner{IC0GPU}(A::AbstractSparseArray{F,I}) where {F,I} = begin
    backend = get_backend(A)
    m, n = size(A)
    m == n || throw("Matrix not square")
    # S = _convert_array!(zeros(m), backend)
    S = zero(I)
    P = KP.kp_ic0(A)
    Preconditioner{IC0GPU,typeof(A),typeof(P),typeof(S)}(A,P,S)
end

Preconditioner{ILU0GPU}(A::AbstractSparseArray{F,I}) where {F,I} = begin
    backend = get_backend(A)
    m, n = size(A)
    m == n || throw("Matrix not square")
    # S = _convert_array!(zeros(m), backend)
    S = zero(I)
    P = KP.kp_ic0(A)
    Preconditioner{ILU0GPU,typeof(A),typeof(P),typeof(S)}(A,P,S)
end

Preconditioner{Jacobi}(A::AbstractSparseArray{F,I}) where {F,I} = begin
    backend = get_backend(A)
    m, n = size(A)
    m == n || throw("Matrix not square")
    S = _convert_array!(zeros(m), backend)
    P = opDiagonal(S)
    Preconditioner{Jacobi,typeof(A),typeof(P),typeof(S)}(A,P,S)
end

# Preconditioner{LDL}(A::AbstractSparseArray{F,I}) where {F,I} = begin
#     m, n = size(A)
#     m == n || throw("Matrix not square")
#     S = zeros(F, m)
#     # P = similar(A)
#     # triu!(P)
#     # P  = opLDL(P)
#     # # P  = opLDL(P)
#     P  = opLDL(A)
#     Preconditioner{LDL,typeof(A),typeof(P),typeof(S)}(A,P,S)
# end

# Preconditioner{ILU0}(A::AbstractSparseArray{F,I}) where {F,I} = begin
#     m, n = size(A)
#     m == n || throw("Matrix not square")
#     S = ilu0(A)
#     P  = LinearOperator(
#         F, m, n, false, false, (y, v) -> ldiv!(y, S, v)
#         )
#     Preconditioner{ILU0,typeof(A),typeof(P),typeof(S)}(A,P,S)
# end

struct DILUprecon{M,V,VI}
    A::M
    D::V
    Di::VI
end
Adapt.@adapt_structure DILUprecon

# Preconditioner{DILU}(A::AbstractSparseArray{F,I}) where {F,I} = begin
#     m, n = size(A)
#     m == n || throw("Matrix not square")
#     D = zeros(F, m)
#     Di = zeros(I, m)
#     diagonal_indices!(Di, A)
#     S = DILUprecon(A, D, Di)
#     P  = LinearOperator(
#         F, m, n, false, false, (y, v) -> ldiv!(y, S, v)
#         )
#     Preconditioner{DILU,typeof(A),typeof(P),typeof(S)}(A,P,S)
# end

Preconditioner{DILU}(A::SparseMatrixCSR{N,F,I}) where {N,F,I} = begin
    m, n = size(A)
    m == n || throw("Matrix not square")
    D = zeros(F, m)
    Di = zeros(I, m)
    diagonal_indices!(Di, A)
    S = DILUprecon(A, D, Di)
    P = S
    Preconditioner{DILU,typeof(A),typeof(P),typeof(S)}(A,P,S)
end
