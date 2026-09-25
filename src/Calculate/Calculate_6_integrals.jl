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

_unit_weight(x, y, z) = one(x)

"""
    volume_integral(phi::ScalarField, config) → Tf
    volume_integral(phi::VectorField, config) → Vector{3}

Volume integral of a field: `∫ phi dV` (component-wise for vector fields).
Runs on the field's backend (CPU or GPU).
"""
volume_integral(phi, config) = weighted_volume_integral(phi, _unit_weight, config)

"""
    weighted_volume_integral(phi::ScalarField, weight_func, config) → Tf
    weighted_volume_integral(phi::VectorField, weight_func, config) → Vector{3}

`∫ phi(x) * w(x, y, z) dV` where `w = weight_func(x, y, z)` is evaluated at each
cell centroid (component-wise for vector fields).  Runs on the field's backend.
For GPU backends, `weight_func` must be callable from a device kernel.
"""
function weighted_volume_integral(phi::ScalarField, weight_func::Func, config) where Func<:Function
    _weighted_sum(phi, weight_func, eltype(phi), config)
end

function weighted_volume_integral(phi::VectorField, weight_func::Func, config) where Func<:Function
    Vector(_weighted_sum(phi, weight_func, SVector{3,eltype(phi.x)}, config))
end

function _weighted_sum(phi, weight_func::Func, T, config) where Func
    (; hardware) = config
    (; backend, workgroup) = hardware
    cells    = phi.mesh.cells
    n        = length(cells)
    products = KA.zeros(backend, T, n)
    kernel!  = _weighted_products!(_setup(backend, workgroup, n)...)
    kernel!(products, phi, cells, weight_func)
    KA.synchronize(backend)
    return sum(products)
end

@kernel function _weighted_products!(products, phi, cells, weight_func::Func) where Func
    i = @index(Global)
    @inbounds begin
        (; centre, volume) = cells[i]
        products[i] = phi[i] * weight_func(centre[1], centre[2], centre[3]) * volume
    end
end

"""
    volume_average(phi::ScalarField, config) → Tf
    volume_average(phi::VectorField, config) → Vector{3}

Volume-averaged mean: `(∫ phi dV) / (∫ dV)`.
"""
volume_average(phi, config) = volume_integral(phi, config) ./ total_volume(phi.mesh, config)
