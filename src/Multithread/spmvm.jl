export SparseXCSR 
export activate_multithread

struct SparseXCSR{Bi,Tv,Ti,N} <: AbstractSparseArray{Tv,Ti,N}
    parent::SparseMatrixCSR{Bi,Tv,Ti}
end

SparseXCSR(A::SparseMatrixCSR{Bi,Tv,Ti}) where {Bi,Tv,Ti} = SparseXCSR{Bi,Tv,Ti,2}(A)

Base.parent(A::SparseXCSR) = A.parent
Base.size(A::SparseXCSR) = size(parent(A))
KernelAbstractions.get_backend(A::SparseXCSR) = get_backend(A.parent.nzval)

# NOTE: The code below has been taken from https://github.com/BacAmorim/ThreadedSparseCSR.jl
# ThreadedSparseCSR has not been updated in a while and precompilation fails on Julia 1.11.1

struct RangeIterator
    k::Int
    d::Int
    r::Int
end

RangeIterator(n::Int, k::Int) = RangeIterator(min(n,k),divrem(n,k)...)
Base.length(it::RangeIterator) = it.k
endpos(it::RangeIterator, i::Int) = i*it.d+min(i,it.r)
Base.iterate(it::RangeIterator, i::Int=1) = i>it.k ? nothing : (endpos(it,i-1)+1:endpos(it,i), i+1)


# function xmul!(y::AbstractVector, A::SparseMatrixCSR, x::AbstractVector, alpha::Number, beta::Number)
function xmul!(y::AbstractVector, Ax::SparseXCSR, x::AbstractVector, alpha::Number, beta::Number)
    
    A = parent(Ax)
    A.n == size(x, 1) || throw(DimensionMismatch())
    A.m == size(y, 1) || throw(DimensionMismatch())

    o = getoffset(A)

    @sync for r in RangeIterator(size(y, 1), Threads.nthreads())
        Threads.@spawn for row in r
            @inbounds begin
                accu = zero(eltype(y))
                for nz in nzrange(A, row)
                    col = A.colval[nz] + o
                    accu += A.nzval[nz]*x[col]
                end
                y[row] = alpha*accu + beta*y[row]
            end
        end
    end

    return y

end

# function xmul!(A::SparseMatrixCSR, x::AbstractVector)
#     xmul!(y, A, x, true, false)
# end

function xmul!(A::SparseXCSR, x::AbstractVector)
    xmul!(y, parent(A), x, true, false)
end

# function xmul(y::AbstractVector, A::SparseMatrixCSR, x::AbstractVector)
#     y = similar(x)
#     xmul!(y, A, x, true, false)
# end

function xmul(y::AbstractVector, A::SparseXCSR, x::AbstractVector)
    y = similar(x)
    xmul!(y, parent(A), x, true, false)
end

# Now add methods for Base and LinearAlgebra 

# """
    function activate_multithread(backend::CPU)

        BLAS.set_num_threads(1)

        nothing
    end

function  LinearAlgebra.mul!(y::AbstractVector, A::SparseXCSR, x::AbstractVector, alpha::Number, beta::Number)
    return xmul!(y, A, x, alpha, beta)
end

function  LinearAlgebra.mul!(y::AbstractVector, A::SparseXCSR, x::AbstractVector)
    return xmul!(y, A, x, true, false)
end

function  Base.:*(A::SparseMatrixCSR, x::SparseXCSR)
    return xmul(A, x)
end