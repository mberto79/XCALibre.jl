export radial_mask!

function radial_mask!(x0, radius_inner, radius_outer, hardware, mesh)
    (; backend, workgroup) = hardware
    cells = mesh.cells 
    mask = ScalarField(mesh)

    ndrange = length(cells)
    kernel! = _radial_mask!(_setup(backend, workgroup, ndrange)...)
    kernel!(x0, radius_inner, radius_outer, mask, cells)
end

@kernel function _radial_mask!(x0, radius_inner, radius_outer, mask, cells)
    cID = @index(Global)

    r = cells[cID].centre - x0
    length = norm(r)
    if length <= radius_outer
        if length >= radius_inner
            mask[cID] = 1
        end
    end

end



