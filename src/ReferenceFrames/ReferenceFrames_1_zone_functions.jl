export radial_mask!

radial_mask!(x0, radius_inner, radius_outer, hardware, mesh; ID = 1, mask=nothing) = begin
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

disc_mask!(x0, x1, radius_inner, radius_outer, hardware, mesh; ID = 1, mask=nothing) = begin
    (; backend, workgroup) = hardware
    cells = mesh.cells

    if isnothing(mask)
        mask = ScalarField(mesh)
    end

    ndrange = length(cells)
    kernel! = _disc_mask!(_setup(backend, workgroup, ndrange)...)
    kernel!(x0, x1, radius_inner, radius_outer, mask, cells, ID)
    return mask
end

@kernel function _disc_mask!(x0, x1, radius_inner, radius_outer, mask, cells, ID)
    cID = @index(Global)

    r = cells[cID].centre - x0
    S = x1 - x0

    dot_rS = r ⋅ S
    dot_SS = S ⋅ S
    d_squared = (r ⋅ r) - (dot_Rs^2 / dot_SS)

    if dot_rS < 0 && dot_SS < dot_rS
        if d_squared < radius_outer && d_squared > radius_inner
            mask[cID] = ID
        end
    end
end