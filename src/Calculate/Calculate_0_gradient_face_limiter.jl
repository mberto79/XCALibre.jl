# Face-based gradient limiter (cf. OpenFOAM faceLimitedGrad): per internal face, owner/neighbour bounds are relaxed by ±(max-min)
# (OpenFOAM k=0.5); each cell keeps the minimum limiter ratio over its faces, then a second kernel scales the gradient.
# Unlike CellBased (strict full-neighbourhood bounds), limiter contributions accumulate from both sides of each face.

export limit_gradient!
export FaceBased

struct FaceBased{T}
    limiter::T
end
Adapt.@adapt_structure FaceBased

FaceBased(mesh::AbstractMesh) = begin
    backend = _get_backend(mesh)
    F = _get_float(mesh)
    limiter = fill!(allocate(backend, F, length(mesh.cells)), one(F))
    FaceBased(limiter)
end

### GRADIENT LIMITER - EXPERIMENTAL

function limit_gradient!(method::FaceBased, ∇F, F::AbstractField, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    (; cells, faces, boundary_cellsID,) = F.mesh

    limiter = method.limiter
    limiter .= one(eltype(limiter))

    nbfaces = length(boundary_cellsID)
    internal_faces = length(faces) - nbfaces

    ndrange = internal_faces
    kernel! = _sized(_limit_gradient!, backend, workgroup, ndrange)
    kernel!(method, limiter, ∇F, F, cells, faces, nbfaces)

    ndrange = length(F)
    kernel! = _sized(_update_gradient!, backend, workgroup, ndrange)
    kernel!(∇F, limiter)

    sync!(∇F.result, F.mesh, config) # ghost limiter values are wrong (partial face lists)
    nothing
end

@kernel function _update_gradient!(grad, limiter)
    cID = @index(Global)

    @inbounds begin
        grad.result[cID] *= limiter[cID]
    end
end

@kernel function _limit_gradient!(::FaceBased, limiter, ∇F, F::Field, cells, faces, nbfaces
    ) where {Field<:AbstractScalarField}
    i = @index(Global)
    fID = i + nbfaces

    @inbounds begin
        face = faces[fID]
        ownerCells = face.ownerCells
        owner1 = ownerCells[1]
        owner2 = ownerCells[2]
        cell1 = cells[owner1]
        cell2 = cells[owner2]

        cf = face.centre
        c1 = cell1.centre
        c2 = cell2.centre
        d1 = (cf - c1)
        d2 = (cf - c2)

        F1 = F[owner1]
        F2 = F[owner2]
        grad1 = ∇F[owner1]
        grad2 = ∇F[owner2]

        minF = min(F1, F2)
        maxF = max(F1, F2)
        deltaF = (maxF - minF)
        minF -= deltaF
        maxF += deltaF

        F1_ext = grad1⋅d1
        F2_ext = grad2⋅d2

        set_limiter(limiter, owner1, maxF, minF, F1_ext, F1)
        set_limiter(limiter, owner2, maxF, minF, F2_ext, F2)
    end
end

# Vector kernel follows OpenFOAM faceLimitedGrad<vector>: the gradient tensor extrapolated to the face centre (tensor*d) sets a direction,
# both cells' values are projected onto it (dot products), and bounds and limiter ratios are computed in that scalar space
# using magSqr(gradf).
@kernel function _limit_gradient!(::FaceBased, limiter, ∇F, F::Field, cells, faces, nbfaces
    ) where {Field<:AbstractVectorField}
    i = @index(Global)
    fID = i + nbfaces

    @inbounds begin
        face = faces[fID]
        ownerCells = face.ownerCells
        own = ownerCells[1]
        nei = ownerCells[2]

        cf = face.centre
        cOwn = cells[own].centre
        cNei = cells[nei].centre

        gradOwn = ∇F[own]   # SMatrix{3,3} — gradient tensor
        gradNei = ∇F[nei]
        fOwn = F[own]        # SVector{3} — field value
        fNei = F[nei]

        # --- Owner side ---
        # Extrapolate owner gradient to face centre (tensor × vector → vector)
        gradf = gradOwn * (cf - cOwn)
        # Project both field values onto extrapolation direction (vector ⋅ vector → scalar)
        vsfOwn = gradf ⋅ fOwn
        vsfNei = gradf ⋅ fNei
        # Relaxed bounds in projected space
        maxFace = max(vsfOwn, vsfNei)
        minFace = min(vsfOwn, vsfNei)
        deltaF = maxFace - minFace
        maxFace += deltaF
        minFace -= deltaF
        # Limit using magSqr(gradf) as the extrapolation measure
        set_limiter(limiter, own, maxFace, minFace, gradf ⋅ gradf, vsfOwn)

        # --- Neighbour side ---
        gradf = gradNei * (cf - cNei)
        vsfOwn = gradf ⋅ fOwn
        vsfNei = gradf ⋅ fNei
        maxFace = max(vsfOwn, vsfNei)
        minFace = min(vsfOwn, vsfNei)
        deltaF = maxFace - minFace
        maxFace += deltaF
        minFace -= deltaF
        set_limiter(limiter, nei, maxFace, minFace, gradf ⋅ gradf, vsfNei)
    end
end

function atomic_min!(limiter, i, val)
    old = limiter[i]
    while val < old
        old, success = Atomix.@atomicreplace limiter[i] old => val
        !success || break
    end
    return nothing
end

function set_limiter(limiter, cID, Fmax, Fmin, δF, F)
    δmax = Fmax - F
    δmin = Fmin - F
    if δF > δmax
        atomic_min!(limiter, cID, δmax/δF)
    elseif δF < δmin
        atomic_min!(limiter, cID, δmin/δF)
    end
end
