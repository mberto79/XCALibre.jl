function adjust_boundary!(b_cpu, BC::Union{PeriodicParent,Periodic}, phif::FaceScalarField, phi, boundaries, boundary_cellsID, time,  backend, workgroup)
    phif_values = phif.values
    phi_values = phi.values

    (; faces) = phif.mesh
    face_map = BC.value.face_map

    kernel_range = length(b_cpu[BC.ID].IDs_range)

    kernel! = adjust_boundary_periodic_scalar!(backend, workgroup)
    kernel!(BC, phif, phi, boundaries, boundary_cellsID, time, face_map, faces, phif_values, phi_values, ndrange = kernel_range)
    # # KernelAbstractions.synchronize(backend)
end

@kernel function adjust_boundary_periodic_scalar!(BC, phif, phi, boundaries, boundary_cellsID, time, face_map, faces, phif_values, phi_values)
    i = @index(Global)
    mesh = phi.mesh
    (; cells) = mesh

    @inbounds begin
        pfID = BC.value.face_map[i] # id of periodic face
        pface = faces[pfID]
        pcID = pface.ownerCells[1]
        # pcell = cells[pcID]
        (; IDs_range) = boundaries[BC.ID]
        fID = IDs_range[i]
        face = faces[fID]
        cID = boundary_cellsID[fID]

        # xf = faces[fID].centre
        # xC = cells[cID].centre
        # xN = cells[pcID].centre # probably needs translating by distance between patches!!

        delta1 = face.delta #*norm(face.e ⋅ face.normal)
        delta2 = pface.delta #*norm(pface.e ⋅ pface.normal)
        delta = delta1 + delta2
        
        # Calculate weights using normal functions
        # weight = norm(xf - xC)/norm(xN - xC)
        weight = delta2/delta
        # weight = norm(xf - xC)/(norm(xN - xC) - BC.value.distance)
        one_minus_weight = one(eltype(weight)) - weight

        # phif_values[fID] = 0.5*(phi_values[cID] + phi_values[pcID]) # linear interpolation 
        phif_values[fID] = weight*phi_values[cID] + one_minus_weight*phi_values[pcID]
    end
end

function adjust_boundary!(b_cpu, BC::Union{PeriodicParent,Periodic}, psif::FaceVectorField, psi::VectorField, boundaries, boundary_cellsID, time, backend, workgroup)
    (; x, y, z) = psif

    (; faces) = psif.mesh
    face_map = BC.value.face_map

    kernel_range = length(b_cpu[BC.ID].IDs_range)

    kernel! = adjust_boundary_periodic_vector!(backend, workgroup)
    kernel!(BC, psif, psi, boundaries, boundary_cellsID, time, face_map, faces, x, y, z, ndrange = kernel_range)
    # # KernelAbstractions.synchronize(backend)
end

@kernel function adjust_boundary_periodic_vector!(BC, psif, psi, boundaries, boundary_cellsID, time, face_map, faces, x, y, z)
    i = @index(Global)

    @inbounds begin
        pfID = BC.value.face_map[i] # id of periodic face
        pface = faces[pfID]
        pcID = pface.ownerCells[1]
        (; IDs_range) = boundaries[BC.ID]
        fID = IDs_range[i]
        face = faces[fID]
        cID = boundary_cellsID[fID]

        # w = 0.5

        delta1 = face.delta #*norm(face.e ⋅ face.normal)
        delta2 = pface.delta #*norm(pface.e ⋅ pface.normal)
        delta = delta1 + delta2
        w = delta2/delta
        # psi_face = 0.5*(psi[cID] + psi[pcID]) # linear interpolation 
        psi_face = w*psi[cID] + (1 - w)psi[pcID] # linear interpolation 
        x[fID] = psi_face[1]
        y[fID] = psi_face[2]
        z[fID] = psi_face[3]
    end
end