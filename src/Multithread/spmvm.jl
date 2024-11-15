export activate_multithread1 

function xmul!(y::AbstractVector, A::SparseMatrixCSR, x::AbstractVector, alpha::Number, beta::Number)
    backend = get_backend(x)
    # workgroup = 32
    workgroup = cld(length(x), Threads.nthreads())
    kernel = _xmul!(backend, workgroup)
    kernel(y, A, x, alpha, beta, ndrange=length(x))
    KernelAbstractions.synchronize(backend)
    return y
end

@kernel function _xmul!(y, A, x, alpha, beta)
    i = @index(Global)

    @uniform begin
        cols = colvals(A)
        nzval = nonzeros(A)
    end
    
    @inbounds begin
        acc = zero(eltype(x))
        for nzi ∈ nzrange(A, i)
            j = cols[nzi]
            acc += nzval[nzi]*x[j]
        end
        y[i] = alpha*acc + beta*y[i]
    end
end

function xmul!(A::SparseMatrixCSR, x::AbstractVector)
    xmul!(y, A, x, true, false)
end

function xmul(y::AbstractVector, A::SparseMatrixCSR, x::AbstractVector)
    y = similar(x)
    xmul!(y, A, x, true, false)
end

# Now add methods for Base and LinearAlgebra 

function activate_multithread1()

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
