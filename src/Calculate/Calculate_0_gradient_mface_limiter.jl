export limit_gradient!
export MFaceBased

struct MFaceBased end
MFaceBased(mesh::AbstractMesh) = MFaceBased()

### GRADIENT LIMITER - EXPERIMENTAL

function limit_gradient!(method::MFaceBased, ∇F, F, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    (; cells, faces, cell_faces, cell_neighbours) = F.mesh

    ndrange = length(cells)
    kernel! = _sized(_limit_gradient!, backend, workgroup, ndrange)
    kernel!(method, ∇F, F, cells, faces, cell_faces, cell_neighbours)
    sync!(∇F.result, F.mesh, config) # ghost updates depend on faces absent locally
end

# One work item per cell, visiting its internal faces in order: only this cell's gradient
# is written, so the successive face corrections cannot race with those of its neighbours.
@kernel function _limit_gradient!(
    method::MFaceBased, ∇F, F, cells, faces, cell_faces, cell_neighbours)
    cID = @index(Global)

    @inbounds begin
        c = cells.centre[cID]
        FP = F[cID]
        for fi ∈ cells.faces_range[cID]
            fID = cell_faces[fi]
            FN = F[cell_neighbours[fi]]

            minF = min(FP, FN)
            maxF = max(FP, FN)
            d = faces.centre[fID] - c

            set_limiter(method, ∇F, cID, maxF - FP, minF - FP, d)
        end
    end
end

function set_limiter(
    ::MFaceBased, ∇F::Grad{S,F,R,I,M}, cID, δmax, δmin, d
    ) where {S,F,R<:VectorField,I,M}
    gradP = ∇F[cID]
    fval = gradP⋅d
    d2 = d⋅d

    if fval > δmax
        ∇F.result[cID] = gradP + d*(δmax - fval)/(d2)
    elseif fval < δmin
        ∇F.result[cID] = gradP + d*(δmin - fval)/(d2)
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
    d2 = d⋅d

    if fvalx > δmax[1]
        gradPx = gradPx + d*(δmax[1] - fvalx)/(d2)
    elseif fvalx < δmin[1]
        gradPx = gradPx + d*(δmin[1] - fvalx)/(d2)
    end

    if fvaly > δmax[2]
        gradPy = gradPy + d*(δmax[2] - fvaly)/(d2)
    elseif fvaly < δmin[2]
        gradPy = gradPy + d*(δmin[2] - fvaly)/(d2)
    end

    if fvalz > δmax[3]
        gradPz = gradPz + d*(δmax[3] - fvalz)/(d2)
    elseif fvalz < δmin[3]
        gradPz = gradPz + d*(δmin[3] - fvalz)/(d2)
    end

    ∇F.result[cID] = SMatrix{3,3}(
        gradPx[1], gradPy[1], gradPz[1],
        gradPx[2], gradPy[2], gradPz[2],
        gradPx[3], gradPy[3], gradPz[3],
        )
    nothing
end  