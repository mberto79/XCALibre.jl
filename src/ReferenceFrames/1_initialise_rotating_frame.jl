export initialise_srf!


function initialise_srf!(U,  x0, rotaxis, omega, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    mesh = U.mesh
    cells = mesh.cells 

    ndrange = length(cells)
    kernel! = _initialise_srf!(_setup(backend, workgroup, ndrange)...)
    kernel!(U,  x0, rotaxis, omega, cells)
end

@kernel function _initialise_srf!(U,  x0, rotaxis, omega, cells)
    cID = @index(Global)

    Omega = omega*rotaxis
    r = cells[cID].centre - x0
    U[cID] = U[cID] - (Omega × r)    # Beware the sign
end