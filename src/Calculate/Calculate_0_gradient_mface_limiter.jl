export limit_gradient!
export MFaceBased

struct MFaceBased end

### GRADIENT LIMITER - EXPERIMENTAL

function limit_gradient!(method::MFaceBased, ∇F, F, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    (; cells, faces, boundary_cellsID,) = F.mesh

    nbfaces = length(boundary_cellsID)
    internal_faces = length(faces) - nbfaces

    kernel! = _limit_gradient!(backend, workgroup)
    kernel!(method, ∇F, F, cells, faces, nbfaces, ndrange=internal_faces)
    KernelAbstractions.synchronize(backend)

end

@kernel function _limit_gradient!(method::MFaceBased, ∇F, F, cells, faces, nbfaces)
    i = @index(Global)
    fID = i + nbfaces

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
    # grad1 = ∇F[owner1]
    # grad2 = ∇F[owner2]

    minF = min(F1, F2)
    maxF = max(F1, F2)
    # deltaF = (maxF - minF)
    # minF -= deltaF
    # maxF += deltaF
    # F1_ext = d1⋅grad1
    # F2_ext = d2⋅grad2
    F1_ext = d1
    F2_ext = d2

    set_limiter(method, ∇F, owner1, maxF - F1, minF - F1, F1_ext)
    set_limiter(method, ∇F, owner2, maxF - F2, minF - F2, F2_ext)
end

function set_limiter(
    ::MFaceBased, ∇F::Grad{S,F,R,I,M}, cID, δmax, δmin, d
    ) where {S,F,R<:VectorField,I,M}
    gradP = ∇F[cID]
    fval = gradP⋅d

    if fval > δmax
        ∇F.result[cID] = gradP + d*(δmax - fval)/(d⋅d)
    elseif fval < δmin
        ∇F.result[cID] = gradP + d*(δmin - fval)/(d⋅d)
    end
end  

function set_limiter(
    ::MFaceBased, ∇F::Grad{S,F,R,I,M}, cID, δmax, δmin, d
    ) where {S,F,R<:TensorField,I,M}
    gradP = ∇F[cID]
    z = zero(eltype(gradP))
    res = SMatrix{3,3}(z,z,z,z,z,z,z,z,z)
    gradPx = gradP[1,:]
    gradPy = gradP[2,:]
    gradPz = gradP[3,:]
    fvalx = gradPx⋅d
    fvaly = gradPy⋅d
    fvalz = gradPz⋅d

    if fvalx > δmax[1]
        gradPx = gradPx + d*(δmax[1] - fvalx)/(d⋅d)
    elseif fvalx < δmin[1]
        gradPx = gradPx + d*(δmin[1] - fvalx)/(d⋅d)
    end

    if fvaly > δmax[2]
        gradPy = gradPy + d*(δmax[2] - fvaly)/(d⋅d)
    elseif fvaly < δmin[2]
        gradPy = gradPy + d*(δmin[2] - fvaly)/(d⋅d)
    end

    if fvalz > δmax[3]
        gradPz = gradPz + d*(δmax[3] - fvalz)/(d⋅d)
    elseif fvalz < δmin[3]
        gradPz = gradPz + d*(δmin[3] - fvalz)/(d⋅d)
    end

    ∇F.result[cID] = SMatrix{3,3}(
        gradPx[1], gradPy[1], gradPz[1],
        gradPx[2], gradPy[2], gradPz[2],
        gradPx[3], gradPy[3], gradPz[3],
        )
    nothing
end  