export correct_boundaries!

function correct_boundaries!(phif, phi, BCs, time, config)
    (; mesh) = phif
    (; boundary_cellsID, boundaries) = mesh 
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(boundary_cellsID)
    kernel! = _correct_boundaries!(backend, workgroup, ndrange)
    kernel!(BCs, phif, phi, boundaries, boundary_cellsID, time)
end

@kernel function _correct_boundaries!(BCs, phif, phi, boundaries, boundary_cellsID, time)
    fID = @index(Global)
    @inbounds adjust_boundaries!(BCs, phif, phi, boundaries, boundary_cellsID, time, fID)
end

@generated function adjust_boundaries!(
    BCs, phif, phi, boundaries, boundary_cellsID, time, fID)
    unpacked_BCs = []
    for bci ∈ 1:length(BCs.parameters)
        unpack = quote
            BC = BCs[$bci]
            (; start, stop) = BC.IDs_range
                if start <= fID <= stop
                    boundary_interpolation!(
                        BC, phif, phi, boundary_cellsID, time, fID)
                end
        end
        push!(unpacked_BCs, unpack)
    end
    quote
        $(unpacked_BCs...)
        nothing 
    end
end