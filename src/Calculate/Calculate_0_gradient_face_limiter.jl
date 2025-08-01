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

function limit_gradient!(method::FaceBased, ∇F, F::AbstractField)
    (; hardware) = get_configuration(CONFIG)
    (; backend, workgroup) = hardware

    (; cells, faces, boundary_cellsID,) = F.mesh

    # limiter = fill!(allocate(backend, eltype(F), length(cells)), one(eltype(F)))
    limiter = method.limiter
    limiter .= one(eltype(limiter))

    nbfaces = length(boundary_cellsID)
    internal_faces = length(faces) - nbfaces

    ndrange = internal_faces
    kernel! = _limit_gradient!(_setup(backend, workgroup, ndrange)...)
    kernel!(method, limiter, ∇F, F, cells, faces, nbfaces)
    # KernelAbstractions.synchronize(backend)

    ndrange = length(F)
    kernel! = _update_gradient!(_setup(backend, workgroup, ndrange)...)
    kernel!(∇F, limiter)

    # KernelAbstractions.synchronize(backend)
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

    face = faces[fID]
    ownerCells = face.ownerCells
    owner1 = ownerCells[1]
    owner2 = ownerCells[2]
    cell1 = cells[owner1]
    cell2 = cells[owner2]

    n = face.normal
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

    set_limiter(limiter, owner1, maxF, minF, F1_ext, F1, F2)
    set_limiter(limiter, owner2, maxF, minF, F2_ext, F2, F1)
end

@kernel function _limit_gradient!(::FaceBased, limiter, ∇F, F::Field, cells, faces, nbfaces
    ) where {Field<:AbstractVectorField}
    i = @index(Global)
    fID = i + nbfaces

    face = faces[fID]
    ownerCells = face.ownerCells
    owner1 = ownerCells[1]
    owner2 = ownerCells[2]
    cell1 = cells[owner1]
    cell2 = cells[owner2]

    n = face.normal
    cf = face.centre 
    c1 = cell1.centre
    c2 = cell2.centre
    d1 = (cf - c1)
    d2 = (cf - c2)
    grad1 = ∇F[owner1]
    grad2 = ∇F[owner2]

    gradf = grad1*d1
    F1 = gradf⋅F[owner1]
    F2 = gradf⋅F[owner2]

    minF = min(F1, F2)
    maxF = max(F1, F2)
    deltaF = (maxF - minF)
    minF -= deltaF
    maxF += deltaF

    F1_ext = gradf⋅gradf
    set_limiter(limiter, owner1, maxF, minF, F1_ext, F1, F2)
    
    gradf = grad2*d2
    F1 = gradf⋅F[owner1]
    F2 = gradf⋅F[owner2]

    minF = min(F1, F2)
    maxF = max(F1, F2)
    deltaF = (maxF - minF)
    minF -= deltaF
    maxF += deltaF
    
    F2_ext = gradf⋅gradf

    set_limiter(limiter, owner2, maxF, minF, F2_ext, F2, F1)
end

function set_limiter(limiter, cID, Fmax, Fmin, δF, F, FN)
    δmax = Fmax - F
    δmin = Fmin - F
    if δF > δmax
        limiter[cID] = min(limiter[cID], δmax/δF)
    elseif δF < δmin
        limiter[cID] = min(limiter[cID], δmin/δF)
    end
end  