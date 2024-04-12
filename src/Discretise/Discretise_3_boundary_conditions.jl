export dirichlet, neumann

# TRANSIENT TERM 
@inline (bc::AbstractBoundary)( # Used for all schemes (using "T")
    term::Operator{F,P,I,Time{T}}, 
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione
    ) where {F,P,I,T} = begin
    nothing
end

# LAPLACIAN TERM (NON-UNIFORM)

@inline (bc::Dirichlet)(
    term::Operator{F,P,I,Laplacian{Linear}}, 
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione
    ) where {F,P,I} = begin
    J = term.flux[fID]
    (; area, delta) = face 
    flux = J*area/delta
    ap = term.sign[1]*(-flux)
    # A[cellID,cellID] += ap
    nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)
    Atomix.@atomic nzval[nIndex] += ap
    Atomix.@atomic b[cellID] += ap*bc.value
    nothing
end

@inline (bc::Neumann)(
    term::Operator{F,P,I,Laplacian{Linear}}, 
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione) where {F,P,I} = begin
    phi = term.phi 
    # # values = phi.values
    # fzero = zero(eltype(b))
    # A[cellID,cellID] += fzero
    # b[cellID] += fzero

    # values = phi.values
    # J = term.flux[fID]
    # (; area, delta) = face 
    # flux = J*area/delta
    # ap = term.sign[1]*(-flux)
    # # A[cellID,cellID] += ap
    # nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)
    # Atomix.@atomic nzval[nIndex] += ap
    # Atomix.@atomic b[cellID] += ap*values[cellID]
    nothing
end

@inline (bc::KWallFunction)(
    term::Operator{F,P,I,Laplacian{T}}, 
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione) where {F,P,I,T}  = begin
    nothing
end

@inline (bc::OmegaWallFunction)(
    term::Operator{F,P,I,Laplacian{T}}, 
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione) where {F,P,I,T} = begin
    nothing # should this be Dirichlet?
end

# DIVERGENCE TERM (NON-UNIFORM)

@inline (bc::Dirichlet)(
    term::Operator{F,P,I,Divergence{Linear}}, 
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione) where {F,P,I} = begin
    # A[cellID,cellID] += 0.0 
    Atomix.@atomic b[cellID] += term.sign[1]*(-term.flux[fID]*bc.value)
    nothing
end

@inline (bc::Neumann)(
    term::Operator{F,P,I,Divergence{Linear}}, 
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione) where {F,P,I} = begin
    ap = term.sign[1]*(term.flux[fID])
    # A[cellID,cellID] += ap
    nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)
    Atomix.@atomic nzval[nIndex] += ap
    nothing
end

@inline (bc::KWallFunction)(
    term::Operator{F,P,I,Divergence{Linear}}, # use upwind for all
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione) where {F,P,I} = begin
    ap = term.sign[1]*(term.flux[fID])
    # A[cellID,cellID] += ap
    nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)
    Atomix.@atomic nzval[nIndex] += ap
    nothing
end

@inline (bc::OmegaWallFunction)(
    term::Operator{F,P,I,Divergence{Linear}}, # might need to change this!!!!
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione) where {F,P,I}  = begin
    ap = term.sign[1]*(term.flux[fID])
    # A[cellID,cellID] += ap
    nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)
    Atomix.@atomic nzval[nIndex] += ap
    nothing
end

@inline (bc::Dirichlet)(
    term::Operator{F,P,I,Divergence{Upwind}}, 
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione) where {F,P,I} = begin
    # A[cellID,cellID] += 0.0 
    # b[cellID] += term.sign[1]*(-term.flux[fID]*bc.value)

    ap = term.sign[1]*(term.flux[fID])
    # A[cellID,cellID] += max(ap, 0.0)
    # nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)
    # nzval[nIndex] += max(ap, 0.0)
    Atomix.@atomic b[cellID] -= ap*bc.value

    # ap = term.sign[1]*(term.flux[fID])
    # b[cellID] += A[cellID,cellID]*bc.value
    # A[cellID,cellID] += A[cellID,cellID]
    nothing
end

@inline (bc::Neumann)(
    term::Operator{F,P,I,Divergence{Upwind}}, 
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione) where {F,P,I} = begin
    phi = term.phi 
    # ap = term.sign[1]*(term.flux[fID])
    # A[cellID,cellID] += ap
    # ap = term.sign[1]*(term.flux[fID])
    # A[cellID,cellID] += ap
    # b[cellID] -= ap*phi[cellID]
    ap = term.sign[1]*(term.flux[fID])
    # A[cellID,cellID] += max(ap, 0.0)
    nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)
    Atomix.@atomic nzval[nIndex] += max(ap, 0.0)
    Atomix.@atomic b[cellID] += max(-ap*phi[cellID], 0.0)
    nothing
end

@inline (bc::KWallFunction)(
    term::Operator{F,P,I,Divergence{Upwind}},
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione) where {F,P,I} = begin
    ap = term.sign[1]*(term.flux[fID])
    # A[cellID,cellID] += max(ap, 0.0)
    nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)
    Atomix.@atomic nzval[nIndex] += max(ap, 0.0)
    nothing
end

@inline (bc::OmegaWallFunction)(
    term::Operator{F,P,I,Divergence{Upwind}}, # might need to change this!!!!
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione) where {F,P,I}  = begin
    ap = term.sign[1]*(term.flux[fID])
    # A[cellID,cellID] += max(ap, 0.0)
    nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)
    Atomix.@atomic nzval[nIndex] += max(ap, 0.0)
    nothing
end

# IMPLICIT SOURCE

@inline (bc::Dirichlet)(
    term::Operator{F,P,I,Si}, 
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione) where {F,P,I} = begin
    nothing
end

@inline (bc::Neumann)(
    term::Operator{F,P,I,Si}, 
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione) where {F,P,I} = begin
    nothing
end

@inline (bc::KWallFunction)(
    term::Operator{F,P,I,Si}, 
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione) where {F,P,I} = begin
    # phi = term.phi[cellID] 
    # flux = term.sign*term.flux[cellID]
    # b[cellID] += flux*phi*cell.volume 
    nothing
end

@inline (bc::OmegaWallFunction)(
    term::Operator{F,P,I,Si}, 
    rowval, colptr, nzval, b, cellID, cell, face, fID, ione) where {F,P,I} = begin
    phi = term.phi[cellID] 
    flux = term.sign*term.flux[cellID]
    Atomix.@atomic b[cellID] += flux*phi*cell.volume 
    nothing
end


## GPU CODE

export Execute_apply_boundary_condition_kernel!

# TRANSIENT TERM 
function Execute_apply_boundary_condition_kernel!(
    bc::AbstractBoundary, term::Operator{F,P,I,Time{T}}, 
    backend, boundaries, faces, cells,
    boundary_cellsID, ione, rowval, colptr, nzval, b) where {F,P,I,T}
    nothing
end

# LAPLACIAN TERM (NON-UNIFORM)
function Execute_apply_boundary_condition_kernel!(
    bc::Dirichlet, term::Operator{F,P,I,Laplacian{Linear}}, 
    backend, boundaries, faces, cells,
    boundary_cellsID, ione, rowval, colptr, nzval, b) where {F,P,I}
    
    kernel! = Dirichlet_laplacian_linear!(backend)
    kernel!(term, bc, ione, boundaries, faces, cells, boundary_cellsID, rowval, colptr, nzval, b, ndrange = 1)
    KernelAbstractions.synchronize(backend)
end

@kernel function Dirichlet_laplacian_linear!(term, BC, ione, boundaries, faces, cells, boundary_cellsID, rowval, colptr, nzval, b)
    i = @index(Global)
    i = BC.ID
    
    @inbounds begin
        (; IDs_range) = boundaries[i]
        for j ∈ IDs_range
            faceID = j
            cellID = boundary_cellsID[j]
            face = faces[faceID]
            cell = cells[cellID] 
            # (BCs[$bci])(model.terms[$t], A, b, cellID, cell, face, faceID)
            J = term.flux[faceID]
            (; area, delta) = face 
            flux = J*area/delta
            ap = term.sign[1]*(-flux)
            
            # start = colptr[cellID]
            # offset = 0
            # for j in start:length(rowval)
            #     offset += 1
            #     if rowval[j] == cellID
            #         break
            #     end
            # end
            # nIndex = start + offset - ione
            nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)

            nzval[nIndex] += ap
            b[cellID] += ap*BC.value
        end
    end
end

function Execute_apply_boundary_condition_kernel!(
    bc::Neumann, term::Operator{F,P,I,Laplacian{Linear}}, 
    backend, boundaries, faces, cells,
    boundary_cellsID, ione, rowval, colptr, nzval, b) where {F,P,I}
    
    kernel! = Neumann_laplacian_linear!(backend)
    kernel!(term, bc, ione, boundaries, faces, cells, boundary_cellsID, rowval, colptr, nzval, b, ndrange = 1)
    KernelAbstractions.synchronize(backend)
end

@kernel function Neumann_laplacian_linear!(term, BC, ione, boundaries, faces, cells, boundary_cellsID, rowval, colptr, nzval, b)
    i = @index(Global)
    i = BC.ID
    
    @inbounds begin
        (; IDs_range) = boundaries[i]
        for j ∈ IDs_range
            faceID = j
            cellID = boundary_cellsID[j]
            face = faces[faceID]
            cell = cells[cellID] 
            # (BCs[$bci])(model.terms[$t], A, b, cellID, cell, face, faceID)
            # phi = term.phi
        end
    end
end

# DIVERGENCE TERM (NON-UNIFORM)
function Execute_apply_boundary_condition_kernel!(
    bc::Dirichlet, term::Operator{F,P,I,Divergence{Linear}}, 
    backend, boundaries, faces, cells,
    boundary_cellsID, ione, rowval, colptr, nzval, b) where {F,P,I}
    
    kernel! = Dirichlet_divergence_linear!(backend)
    kernel!(term, bc, ione, boundaries, faces, cells, boundary_cellsID, rowval, colptr, nzval, b, ndrange = 1)
    KernelAbstractions.synchronize(backend)
end

@kernel function Dirichlet_divergence_linear!(term, BC, ione, boundaries, faces, cells, boundary_cellsID, rowval, colptr, nzval, b)
    i = @index(Global)
    i = BC.ID
    
    @inbounds begin
        (; IDs_range) = boundaries[i]
        for j ∈ IDs_range
            faceID = j
            cellID = boundary_cellsID[j]
            face = faces[faceID]
            cell = cells[cellID] 
            # (BCs[$bci])(model.terms[$t], A, b, cellID, cell, face, faceID)
            b[cellID] += term.sign[1]*(-term.flux[faceID]*BC.value)
        end
    end
end

function Execute_apply_boundary_condition_kernel!(
    bc::Neumann, term::Operator{F,P,I,Divergence{Linear}}, 
    backend, boundaries, faces, cells,
    boundary_cellsID, ione, rowval, colptr, nzval, b) where {F,P,I}
    
    kernel! = Neumann_divergence_linear!(backend)
    kernel!(term, bc, ione, boundaries, faces, cells, boundary_cellsID, rowval, colptr, nzval, b, ndrange = 1)
    KernelAbstractions.synchronize(backend)
end

@kernel function Neumann_divergence_linear!(term, BC, ione, boundaries, faces, cells, boundary_cellsID, rowval, colptr, nzval, b)
    i = @index(Global)
    i = BC.ID

    @inbounds begin
        (; IDs_range) = boundaries[i]
        for j ∈ IDs_range
            faceID = j
            cellID = boundary_cellsID[j]
            face = faces[faceID]
            cell = cells[cellID] 
            # (BCs[$bci])(model.terms[$t], A, b, cellID, cell, face, faceID)
            ap = term.sign[1]*(term.flux[faceID])

            # start = colptr[cellID]
            # offset = 0
            # for j in start:length(rowval)
            #     offset += 1
            #     if rowval[j] == cellID
            #         break
            #     end
            # end
            # nIndex = start + offset - ione
            nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)

            nzval[nIndex] += ap
        end
    end
end

function Execute_apply_boundary_condition_kernel!(
    bc::Dirichlet, term::Operator{F,P,I,Divergence{Upwind}}, 
    backend, boundaries, faces, cells,
    boundary_cellsID, ione, rowval, colptr, nzval, b) where {F,P,I}
    
    kernel! = Dirichlet_divergence_upwind!(backend)
    kernel!(term, bc, ione, boundaries, faces, cells, boundary_cellsID, rowval, colptr, nzval, b, ndrange = 1)
    KernelAbstractions.synchronize(backend)
end

@kernel function Dirichlet_divergence_upwind!(term, BC, ione, boundaries, faces, cells, boundary_cellsID, rowval, colptr, nzval, b)
    i = @index(Global)
    i = BC.ID
    
    @inbounds begin
        (; IDs_range) = boundaries[i]
        for j ∈ IDs_range
            faceID = j
            cellID = boundary_cellsID[j]
            face = faces[faceID]
            cell = cells[cellID]
            # (BCs[$bci])(model.terms[$t], A, b, cellID, cell, face, faceID)
            # A[cellID,cellID] += 0.0 
            # b[cellID] += term.sign[1]*(-term.flux[fID]*bc.value)

            ap = term.sign[1]*(term.flux[faceID])
            # A[cellID,cellID] += max(ap, 0.0)
            b[cellID] -= ap*BC.value

            # ap = term.sign[1]*(term.flux[fID])
            # b[cellID] += A[cellID,cellID]*bc.value
            # A[cellID,cellID] += A[cellID,cellID]
        end
    end
end

function Execute_apply_boundary_condition_kernel!(
    bc::Neumann, term::Operator{F,P,I,Divergence{Upwind}}, 
    backend, boundaries, faces, cells,
    boundary_cellsID, ione, rowval, colptr, nzval, b) where {F,P,I}
    kernel! = Neumann_divergence_upwind!(backend)
    kernel!(term, bc, ione, boundaries, faces, cells, boundary_cellsID, rowval, colptr, nzval, b, ndrange = 1)
    KernelAbstractions.synchronize(backend)
end

@kernel function Neumann_divergence_upwind!(term, BC, ione, boundaries, faces, cells, boundary_cellsID, rowval, colptr, nzval, b)
    i = @index(Global)
    i = BC.ID
    
    @inbounds begin
    (; IDs_range) = boundaries[i]
        for j ∈ IDs_range
            faceID = j
            cellID = boundary_cellsID[j]
            face = faces[faceID]
            cell = cells[cellID]

            # phi = term.phi 
            # ap = term.sign[1]*(term.flux[fID])
            # A[cellID,cellID] += ap
            # ap = term.sign[1]*(term.flux[fID])
            # A[cellID,cellID] += ap
            # b[cellID] -= ap*phi[cellID]
            ap = term.sign[1]*(term.flux[faceID])

            # start = colptr[cellID]
            # offset = 0
            # for j in start:length(rowval)
            #     offset += 1
            #     if rowval[j] == cellID
            #         break
            #     end
            # end
            # nIndex = start + offset - ione
            nIndex = nzval_index(colptr, rowval, cellID, cellID, ione)

            nzval[nIndex] += ap
            # b[cellID] -= max(ap*phi[cellID], 0.0)
        end
    end
end

# IMPLICIT SOURCE

function Execute_apply_boundary_condition_kernel!(
    bc::Dirichlet, term::Operator{F,P,I,Si}, 
    backend, boundaries, faces, cells,
    boundary_cellsID, ione, rowval, colptr, nzval, b) where {F,P,I}
    nothing
end

function Execute_apply_boundary_condition_kernel!(
    bc::Neumann, term::Operator{F,P,I,Si}, 
    backend, boundaries, faces, cells,
    boundary_cellsID, ione, rowval, colptr, nzval, b) where {F,P,I}
    nothing
end