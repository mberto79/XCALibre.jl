export volume_integral, weighted_volume_integral, volume_average, total_volume

"""
    total_volume(mesh) → Tf

Sum of all cell volumes in the mesh domain.  Runs on the mesh backend.
"""
function total_volume(mesh)
    backend = _get_backend(mesh)
    F       = _get_float(mesh)
    cells   = mesh.cells
    n       = length(cells)
    vols    = KA.zeros(backend, F, n)
    kernel! = _cell_volumes!(_setup(backend, 64, n)...)
    kernel!(vols, cells)
    KA.synchronize(backend)
    return sum(vols)
end

@kernel function _cell_volumes!(vols, cells)
    i = @index(Global)
    @inbounds vols[i] = cells[i].volume
end

"""
    volume_integral(phi::ScalarField) → Tf

Volume-weighted integral of a scalar field: `∫ phi dV`.
Runs on the field's backend (CPU or GPU).
"""
function volume_integral(phi::ScalarField)
    backend  = KA.get_backend(phi)
    F        = eltype(phi)
    cells    = phi.mesh.cells
    n        = length(cells)
    products = KA.zeros(backend, F, n)
    kernel!  = _volume_products_scalar!(_setup(backend, 64, n)...)
    kernel!(products, phi, cells)
    KA.synchronize(backend)
    return sum(products)
end

@kernel function _volume_products_scalar!(products, phi, cells)
    i = @index(Global)
    @inbounds products[i] = phi[i] * cells[i].volume
end

"""
    volume_integral(phi::VectorField) → Vector{3}

Volume-weighted integral of each component: `[∫ux dV, ∫uy dV, ∫uz dV]`.
"""
function volume_integral(phi::VectorField)
    backend = KA.get_backend(phi.x)
    F       = eltype(phi.x)
    cells   = phi.mesh.cells
    n       = length(cells)
    px      = KA.zeros(backend, F, n)
    py      = KA.zeros(backend, F, n)
    pz      = KA.zeros(backend, F, n)
    kernel! = _volume_products_vector!(_setup(backend, 64, n)...)
    kernel!(px, py, pz, phi, cells)
    KA.synchronize(backend)
    return [sum(px), sum(py), sum(pz)]
end

@kernel function _volume_products_vector!(px, py, pz, phi, cells)
    i = @index(Global)
    @inbounds begin
        vol  = cells[i].volume
        v    = phi[i]
        px[i] = v[1] * vol
        py[i] = v[2] * vol
        pz[i] = v[3] * vol
    end
end

"""
    weighted_volume_integral(phi::ScalarField, weight_func) → Tf

`∫ phi(x) * w(x, y, z) dV` where `w = weight_func(x, y, z)` is evaluated at each
cell centroid.  Runs on the field's backend. For GPU backends, `weight_func`
must be callable from a device kernel.
"""
function weighted_volume_integral(phi::ScalarField, weight_func::Func) where Func<:Function
    backend  = KA.get_backend(phi)
    F        = eltype(phi)
    cells    = phi.mesh.cells
    n        = length(cells)
    products = KA.zeros(backend, F, n)
    kernel!  = _weighted_products_scalar!(_setup(backend, 64, n)...)
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
    weighted_volume_integral(phi::VectorField, weight_func) → Vector{3}

`[∫ux*w dV, ∫uy*w dV, ∫uz*w dV]` with `w = weight_func(x, y, z)`.
For GPU backends, `weight_func` must be callable from a device kernel.
"""
function weighted_volume_integral(phi::VectorField, weight_func::Func) where Func<:Function
    backend = KA.get_backend(phi.x)
    F       = eltype(phi.x)
    cells   = phi.mesh.cells
    n       = length(cells)
    px      = KA.zeros(backend, F, n)
    py      = KA.zeros(backend, F, n)
    pz      = KA.zeros(backend, F, n)
    kernel! = _weighted_products_vector!(_setup(backend, 64, n)...)
    kernel!(px, py, pz, phi, cells, weight_func)
    KA.synchronize(backend)
    return [sum(px), sum(py), sum(pz)]
end

@kernel function _weighted_products_vector!(px, py, pz, phi, cells, weight_func::Func) where Func
    i = @index(Global)
    @inbounds begin
        c    = cells[i].centre
        w    = weight_func(c[1], c[2], c[3])
        vol  = cells[i].volume
        v    = phi[i]
        px[i] = v[1] * w * vol
        py[i] = v[2] * w * vol
        pz[i] = v[3] * w * vol
    end
end

"""
    volume_average(phi::ScalarField) → Tf

Volume-averaged mean: `(∫ phi dV) / (∫ dV)`.
"""
function volume_average(phi::ScalarField)
    return volume_integral(phi) / total_volume(phi.mesh)
end

"""
    volume_average(phi::VectorField) → Vector{3}

Volume-averaged mean of a vector field.
"""
function volume_average(phi::VectorField)
    return volume_integral(phi) ./ total_volume(phi.mesh)
end
