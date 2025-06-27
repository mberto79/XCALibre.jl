function delta!(Δ, mesh, config)
    # Extract hardware configuration
    (; hardware) = config
    (; backend, workgroup) = hardware

    (; cells) = mesh

    # set up and launch kernel
    ndrange = length(cells)
    kernel! = _delta!(_setup(backend, workgroup, ndrange)...)
    kernel!(Δ, cells)
    # KernelAbstractions.synchronize(backend)
end

@kernel function _delta!(Δ, cells) # Add types in call to get float type at compile time
    i = @index(Global)

    @uniform begin
        p = 1/3
        values = Δ.values
    end

    @inbounds begin
        values[i] = (cells[i].volume)^p
    end
end