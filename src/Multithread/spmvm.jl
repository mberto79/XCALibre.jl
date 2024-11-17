export activate_multithread

# function xmul!(y::AbstractVector, A::SparseMatrixCSR, x::AbstractVector, alpha::Number, beta::Number)
#     backend = get_backend(x)
#     workgroup = 32
#     # workgroup = cld(length(x), Threads.nthreads())
#     kernel = _xmul!(backend, workgroup)
#     kernel(y, A, x, alpha, beta, ndrange=length(x))
#     KernelAbstractions.synchronize(backend)
#     return y
# end

# @kernel function _xmul!(y, A, x, alpha, beta)
#     i = @index(Global)

#     @uniform begin
#         cols = colvals(A)
#         nzval = nonzeros(A)
#     end
    
#     @inbounds begin
#         acc = zero(eltype(x))
#         for nzi ∈ nzrange(A, i)
#             j = cols[nzi]
#             acc += nzval[nzi]*x[j]
#         end
#         y[i] = alpha*acc + beta*y[i]
#     end
# end

struct RangeIterator
    k::Int
    d::Int
    r::Int
end

RangeIterator(n::Int, k::Int) = RangeIterator(min(n,k),divrem(n,k)...)
Base.length(it::RangeIterator) = it.k
endpos(it::RangeIterator, i::Int) = i*it.d+min(i,it.r)
Base.iterate(it::RangeIterator, i::Int=1) = i>it.k ? nothing : (endpos(it,i-1)+1:endpos(it,i), i+1)


function xmul!(y::AbstractVector, A::SparseMatrixCSR, x::AbstractVector, alpha::Number, beta::Number)
    
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

function xmul!(A::SparseMatrixCSR, x::AbstractVector)
    xmul!(y, A, x, true, false)
end

function xmul(y::AbstractVector, A::SparseMatrixCSR, x::AbstractVector)
    y = similar(x)
    xmul!(y, A, x, true, false)
end

# Now add methods for Base and LinearAlgebra 

"""
    function activate_multithread(backend::CPU)

        BLAS.set_num_threads(1)

        @eval function  mul!(y::AbstractVector, A::SparseMatrixCSR, x::AbstractVector, alpha::Number, beta::Number)
            return xmul!(y, A, x, alpha, beta)
        end

        @eval function  mul!(y::AbstractVector, A::SparseMatrixCSR, x::AbstractVector)
            return xmul!(y, A, x, true, false)
        end

        @eval function  *(A::SparseMatrixCSR, x::AbstractVector)
            return xmul(A, x)
        end

        nothing
    end

Function to activate multithreading for CSR sparse matrices. The only input required is the backend (which must be `CPU()`).

"""
function activate_multithread(backend::CPU)

    BLAS.set_num_threads(1)

    @eval function  mul!(y::AbstractVector, A::SparseMatrixCSR, x::AbstractVector, alpha::Number, beta::Number)
        return xmul!(y, A, x, alpha, beta)
    end

    @eval function  mul!(y::AbstractVector, A::SparseMatrixCSR, x::AbstractVector)
        return xmul!(y, A, x, true, false)
    end

    @eval function  *(A::SparseMatrixCSR, x::AbstractVector)
        return xmul(A, x)
    end

    nothing
end
