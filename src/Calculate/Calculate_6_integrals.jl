export volume_integral, weighted_volume_integral, volume_average, total_volume

"""
    total_volume(mesh, config) → Tf

Sum of all cell volumes in the mesh domain.  Runs on the mesh backend.
"""
function total_volume(mesh, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    F       = _get_float(mesh)
    cells   = mesh.cells
    n       = length(cells)
    vols    = KA.zeros(backend, F, n)
    kernel! = _cell_volumes!(_setup(backend, workgroup, n)...)
    kernel!(vols, cells)
    KA.synchronize(backend)
    return sum(vols)
end

@kernel function _cell_volumes!(vols, cells)
    i = @index(Global)
    @inbounds vols[i] = cells[i].volume
end

"""
    volume_integral(phi::ScalarField, config) → Tf

Volume-weighted integral of a scalar field: `∫ phi dV`.
Runs on the field's backend (CPU or GPU).
"""
function volume_integral(phi::ScalarField, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    F        = eltype(phi)
    cells    = phi.mesh.cells
    n        = length(cells)
    products = KA.zeros(backend, F, n)
    kernel!  = _volume_products_scalar!(_setup(backend, workgroup, n)...)
    kernel!(products, phi, cells)
    KA.synchronize(backend)
    return sum(products)
end

@kernel function _volume_products_scalar!(products, phi, cells)
    i = @index(Global)
    @inbounds products[i] = phi[i] * cells[i].volume
end

"""
    volume_integral(phi::VectorField, config) → Vector{3}

Volume-weighted integral of each component: `[∫ux dV, ∫uy dV, ∫uz dV]`.
"""
function volume_integral(phi::VectorField, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    F        = eltype(phi.x)
    cells    = phi.mesh.cells
    n        = length(cells)
    products = KA.zeros(backend, SVector{3,F}, n)
    kernel!  = _volume_products_vector!(_setup(backend, workgroup, n)...)
    kernel!(products, phi, cells)
    KA.synchronize(backend)
    return Vector(sum(products))
end

@kernel function _volume_products_vector!(products, phi, cells)
    i = @index(Global)
    @inbounds begin
        vol = cells[i].volume
        products[i] = phi[i] * vol
    end
end

"""
    weighted_volume_integral(phi::ScalarField, weight_func, config) → Tf

`∫ phi(x) * w(x, y, z) dV` where `w = weight_func(x, y, z)` is evaluated at each
cell centroid.  Runs on the field's backend. For GPU backends, `weight_func`
must be callable from a device kernel.
"""
function weighted_volume_integral(phi::ScalarField, weight_func::Func, config) where Func<:Function
    (; hardware) = config
    (; backend, workgroup) = hardware
    F        = eltype(phi)
    cells    = phi.mesh.cells
    n        = length(cells)
    products = KA.zeros(backend, F, n)
    kernel!  = _weighted_products_scalar!(_setup(backend, workgroup, n)...)
    kernel!(products, phi, cells, weight_func)
    KA.synchronize(backend)
    return sum(products)
end

@kernel function _weighted_products_scalar!(products, phi, cells, weight_func::Func) where Func
    i = @index(Global)
    @inbounds begin
        c          = cells[i].centre
        w          = weight_func(c[1], c[2], c[3])
        products[i] = phi[i] * w * cells[i].volume
    end
end

"""
    weighted_volume_integral(phi::VectorField, weight_func, config) → Vector{3}

`[∫ux*w dV, ∫uy*w dV, ∫uz*w dV]` with `w = weight_func(x, y, z)`.
For GPU backends, `weight_func` must be callable from a device kernel.
"""
function weighted_volume_integral(phi::VectorField, weight_func::Func, config) where Func<:Function
    (; hardware) = config
    (; backend, workgroup) = hardware
    F        = eltype(phi.x)
    cells    = phi.mesh.cells
    n        = length(cells)
    products = KA.zeros(backend, SVector{3,F}, n)
    kernel!  = _weighted_products_vector!(_setup(backend, workgroup, n)...)
    kernel!(products, phi, cells, weight_func)
    KA.synchronize(backend)
    return Vector(sum(products))
end

@kernel function _weighted_products_vector!(products, phi, cells, weight_func::Func) where Func
    i = @index(Global)
    @inbounds begin
        c = cells[i].centre
        w = weight_func(c[1], c[2], c[3])
        products[i] = phi[i] * w * cells[i].volume
    end
end

"""
    volume_average(phi::ScalarField, config) → Tf

Volume-averaged mean: `(∫ phi dV) / (∫ dV)`.
"""
function volume_average(phi::ScalarField, config)
    return volume_integral(phi, config) / total_volume(phi.mesh, config)
end

"""
    volume_average(phi::VectorField, config) → Vector{3}

Volume-averaged mean of a vector field.
"""
function volume_average(phi::VectorField, config)
    return volume_integral(phi, config) ./ total_volume(phi.mesh, config)
end
