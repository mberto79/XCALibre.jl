struct JacobiSmoother{D,L}
    diagonal::D 
    loops::L
end
Adapt.@adapt_structure JacobiSmoother

function apply_smoother!(smoother::JacobiSmoother, x, A, b, mesh, config)
    (; hardware, runtime) = config
    (; backend, workgroup) = hardware

    nzval = _nzval(A)
    colval = _colval(A)
    rowptr = _rowptr(A)

    kernel! = _apply_smoother!(backend, workgroup)
    for _ ∈ 1:smoother.loops
        kernel!(x, b, nzval, colval, rowptr, mesh; ndrange = length(mesh.cells))
        KernelAbstractions.synchronize(backend)
    end
end

@kernel function _apply_smoother!(x, b, nzval, colval, rowptr, mesh)
    cID = @index(Global)
    (; cells, cell_neighbours) = mesh

    cell = cells[cID]
    (; faces_range) = cell
    cIndex = spindex(rowptr, colval, cID, cID)


    sum = zero(_get_float(mesh))

    for nzvali ∈ rowptr[cID]:(rowptr[cID+1] - 1)
        j = colval[nzvali]
        sum += nzval[nzvali]*x[j]
    end
    sum -= nzval[cIndex]*x[cID] # remove multiplication with diagonal (faster than "if")
    rD = one(_get_int(mesh))/nzval[cIndex]
    x[cID] = rD*(b[cID] - sum)

end