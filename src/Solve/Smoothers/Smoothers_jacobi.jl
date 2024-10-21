struct JacobiSmoother{D,L}
    diagonal::D 
    loops::L
end
Adapt.@adapt_structure JacobiSmoother

function apply_smoother!(smoother::JacobiSmoother, x, A, b, mesh, config)
    (; hardware, runtime) = config
    (; backend, workgroup) = hardware

    nzval = _nzval(A)
    rowval = _rowval(A)
    colptr = _colptr(A)

    kernel! = _apply_smoother!(backend, workgroup)
    for _ ∈ 1:smoother.loops
        kernel!(x, b, nzval, rowval, colptr, mesh; ndrange = length(mesh.cells))
        KernelAbstractions.synchronize(backend)
    end
end

@kernel function _apply_smoother!(x, b, nzval, rowval, colptr, mesh)
    cID = @index(Global)
    (; cells, cell_neighbours) = mesh

    cell = cells[cID]
    (; faces_range) = cell
    cIndex = spindex(colptr, rowval, cID, cID)

    # for i ∈ 1:loops
        sum = zero(_get_float(mesh))
        for fi ∈ faces_range
            nID = cell_neighbours[fi]
            nIndex = spindex(colptr, rowval, cID, nID)
            sum += nzval[nIndex]*x[nID]
        end
        rD = one(_get_int(mesh))/nzval[cIndex]
        x[cID] = rD*(b[cID] - sum)
    # end
end