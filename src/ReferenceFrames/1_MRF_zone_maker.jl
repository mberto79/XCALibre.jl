export radial_mask

radial_mask(x0, radius_inner, radius_outer, hardware, mesh; ID = 1, mask=nothing) = begin
    (; backend, workgroup) = hardware
    cells = mesh.cells 
    if isnothing(mask)
        mask = ScalarField(mesh)
    end

    ndrange = length(cells)
    kernel! = _radial_mask!(_setup(backend, workgroup, ndrange)...)
    kernel!(x0, radius_inner, radius_outer, mask, cells, ID)
    return mask
end

@kernel function _radial_mask!(x0, radius_inner, radius_outer, mask, cells, ID)
    cID = @index(Global)

    r = cells[cID].centre - x0
    length = norm(r)
    if length <= radius_outer
        if length >= radius_inner
            mask[cID] = ID
        end
    end

end
