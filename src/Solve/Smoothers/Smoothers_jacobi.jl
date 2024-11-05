struct JacobiSmoother{L,F,V}
    loops::L
    omega::F
    x_temp::V
end
Adapt.@adapt_structure JacobiSmoother

function apply_smoother!(smoother, x, A, b, mesh, config)
    (; hardware, runtime) = config
    (; backend, workgroup) = hardware

    nzval = _nzval(A)
    colval = _colval(A)
    rowptr = _rowptr(A)

    _smoother_launch(backend, smoother, x, b, nzval, colval, rowptr, mesh, workgroup)
end

# CPU implementation
function _smoother_launch(
    backend::CPU, smoother, x, b, nzval, colval, rowptr, mesh, workgroup
    )
    _apply_smoother!(smoother, x, b, nzval, colval, rowptr, mesh)
end

function _apply_smoother!(
    smoother::JacobiSmoother, x, b, nzval, colval, rowptr, mesh
    )
    ω = smoother.omega
    for _ ∈ 1:smoother.loops
        Threads.@threads for cID ∈ eachindex(mesh.cells)
            cIndex =  spindex(rowptr, colval, cID, cID)
            xi = multiply_row(x, b, ω, nzval, rowptr, colval,  cID, cIndex)
            smoother.x_temp[cID] = xi
        end
        Threads.@threads for cID ∈ eachindex(smoother.x_temp)
            @inbounds x[cID] = smoother.x_temp[cID]
        end
    end
end

# GPU Kernel
function _smoother_launch(
    backend::GPU, smoother, x, b, nzval, colval, rowptr, mesh, workgroup
    )
    krange = length(mesh.cells)
    kernel! = _apply_smoother!(backend, workgroup)
    for _ ∈ 1:smoother.loops
        kernel!(smoother, x, b, nzval, colval, rowptr, mesh, ndrange = krange)
        KernelAbstractions.synchronize(backend)
        x .= smoother.x_temp
    end
end

@kernel function _apply_smoother!(
    smoother::JacobiSmoother, x, b, nzval, colval, rowptr, mesh
    )
    cID = @index(Global)
    ω = smoother.omega
    cIndex =  spindex(rowptr, colval, cID, cID)
    xi = multiply_row(x, b, ω, nzval, rowptr, colval,  cID, cIndex)
    smoother.x_temp[cID] = xi
end

# Weighted Jacobi implementation
@inline function multiply_row(x, b, ω, nzval, rowptr, colval,  cID, cIndex)
    sum = zero(eltype(nzval))
    uno = one(eltype(colval))
    @inbounds begin
        for nzvali ∈ rowptr[cID]:(rowptr[cID+1] - 1)
            j = colval[nzvali]
            xj = x[j] 
            sum += nzval[nzvali]*xj
        end
        a_ii = nzval[cIndex]
        xi = x[cID]
        sum -= a_ii*xi # remove multiplication with diagonal (faster than "if")
        rD = one(eltype(colval))/a_ii
        return ω*rD*(b[cID] - sum) + (uno - ω)*xi
    end
end