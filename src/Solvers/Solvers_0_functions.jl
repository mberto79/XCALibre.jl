export flux!, update_nueff!, inverse_diagonal!, remove_pressure_source!, H!, correct_velocity!

## UPDATE VISCOSITY

# This function needs to be separated using multiple dispatch
function update_nueff!(nueff, nu, turb_model, config)
    (; mesh) = nueff
    # backend = _get_backend(mesh)
    (; hardware) = config
    (; backend, workgroup) = hardware

    kernel_range = length(nueff)
    if typeof(turb_model) <: Laminar
        kernel! = update_nueff_laminar!(backend, workgroup, kernel_range)
        kernel!(nu, nueff, ndrange=kernel_range)
    else
        (; nutf) = turb_model
        kernel! = update_nueff_turbulent!(backend, workgroup, kernel_range)
        kernel!(nu, nutf, nueff, ndrange=kernel_range)
    end

end

@kernel function update_nueff_laminar!(nu, nueff)
    i = @index(Global)

    @inbounds begin
        nueff[i] = nu[i]
    end
end

@kernel function update_nueff_turbulent!(nu, νtf, nueff)
    i = @index(Global)

    @inbounds begin
        nueff[i] = nu[i] + νtf[i]
    end
end

## RESIDUAL CALCULATIONS

# Sparse Matrix Multiplication function
function sparse_matmul!(a, b, c, config)
    if size(a)[2] != length(b)
        error("Matrix size mismatch!")
        return nothing
    end

    (; hardware) = config
    (; backend, workgroup) = hardware

    nzval_array = _nzval(a)
    colptr_array = _colptr(a)
    rowval_array = _rowval(a)
    fzero = zero(eltype(c))

    kernel_range = length(c)

    kernel! = matmul_copy_zeros_kernel!(backend, workgroup, kernel_range)
    kernel!(c, fzero, ndrange=kernel_range)
    KernelAbstractions.synchronize(backend)

    kernel! = sparse_matmul_kernel!(backend, workgroup, kernel_range)
    kernel!(nzval_array, rowval_array, colptr_array, b, c, ndrange=kernel_range)
    KernelAbstractions.synchronize(backend)
end

# Sparse Matrix Multiplication kernel
@kernel function sparse_matmul_kernel!(nzval, rowval, colptr, mulvec, res)
    i = @index(Global)
    @inbounds begin
        @synchronize
        start = colptr[i]
        fin = colptr[i+1]

        for j in start:fin-1
            val = nzval[j] #A[j,i]
            row = rowval[j] #Row index of non-zero element in A
            Atomix.@atomic res[row] += mulvec[i] * val
        end
    end
end

# Sparse Matrix Multiplication copy kernel
@kernel function matmul_copy_zeros_kernel!(c, fzero)
    i = @index(Global)

    @inbounds begin
        c[i] = fzero
    end
end

## FLUX CALCULATION

function flux!(phif::FS, psif::FV, config) where {FS<:FaceScalarField,FV<:FaceVectorField}
    (; hardware) = config
    (; backend, workgroup) = hardware

    kernel_range = length(phif)
    kernel! = flux_kernel!(backend, workgroup, kernel_range)
    kernel!(phif, psif, ndrange=kernel_range)
    KernelAbstractions.synchronize(backend)
end

@kernel function flux_kernel!(phif, psif)
    i = @index(Global)

    @uniform begin
        (; mesh, values) = phif
        (; faces) = mesh
    end

    @inbounds begin
        (; area, normal) = faces[i]
        Sf = area * normal
        values[i] = psif[i] ⋅ Sf
    end
end

function flux!(phif::FS, psif::FV, rhof::FS, config) where {FS<:FaceScalarField,FV<:FaceVectorField}
    (; hardware) = config
    (; backend, workgroup) = hardware

    kernel_range = length(phif)
    kernel! = flux_kernel!(backend, workgroup, kernel_range)
    kernel!(phif, psif, rhof, ndrange=kernel_range)
    KernelAbstractions.synchronize(backend)
end

@kernel function flux_kernel!(phif, psif, rhof)
    i = @index(Global)

    @uniform begin
        (; mesh, values) = phif
        (; faces) = mesh
    end

    @inbounds begin
        (; area, normal) = faces[i]
        Sf = area * normal
        values[i] = (psif[i] ⋅ Sf) * rhof[i] 
    end
end


volumes(mesh) = [mesh.cells[i].volume for i ∈ eachindex(mesh.cells)]

# INVERSE DIAGONAL CALCULATION

function inverse_diagonal!(rD::S, eqn, config) where {S<:ScalarField}
    (; hardware) = config
    (; backend, workgroup) = hardware
    A = eqn.equation.A # Or should I use A0
    nzval, rowval, colptr = get_sparse_fields(A)

    kernel_range = length(rD)
    kernel! = inverse_diagonal_kernel!(backend, workgroup, kernel_range)
    kernel!(rD, nzval, rowval, colptr, ndrange=kernel_range)
    KernelAbstractions.synchronize(backend)
end

@kernel function inverse_diagonal_kernel!(rD, nzval, rowval, colptr)
    i = @index(Global)

    @uniform begin
        (; mesh, values) = rD
        cells = mesh.cells
    end

    @inbounds begin
        idx = spindex(colptr, rowval, i, i)
        D = nzval[idx]
        (; volume) = cells[i]
        values[i] = volume / D
    end
end

## VELOCITY CORRECTION

function correct_velocity!(U, Hv, ∇p, rD, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    kernel_range = length(U)
    kernel! = _correct_velocity!(backend, workgroup, kernel_range)
    kernel!(U, Hv, ∇p, rD, ndrange=kernel_range)
    KernelAbstractions.synchronize(backend)
end

@kernel function _correct_velocity!(U, Hv, ∇p, rD)
    i = @index(Global)

    @uniform begin
        Ux, Uy, Uz = U.x, U.y, U.z
        Hvx, Hvy, Hvz = Hv.x, Hv.y, Hv.z
        dpdx, dpdy, dpdz = ∇p.result.x, ∇p.result.y, ∇p.result.z
        rDvalues = rD.values
    end

    @inbounds begin
        rDvalues_i = rDvalues[i]
        Ux[i] = Hvx[i] - dpdx[i] * rDvalues_i
        Uy[i] = Hvy[i] - dpdy[i] * rDvalues_i
        Uz[i] = Hvz[i] - dpdz[i] * rDvalues_i
    end
end

## PRESSURE CORRECTION AND SOURCE REMOVAL

remove_pressure_source!(U_eqn::ME, ∇p, config) where {ME} = begin # Extend to 3D
    # backend = _get_backend(get_phi(ux_eqn).mesh)
    (; hardware) = config
    (; backend, workgroup) = hardware
    cells = get_phi(U_eqn).mesh.cells
    source_sign = get_source_sign(U_eqn, 1)
    (; bx, by, bz) = U_eqn.equation

    kernel_range = length(bx)
    kernel! = remove_pressure_source_kernel!(backend, workgroup, kernel_range)
    kernel!(cells, source_sign, ∇p, bx, by, bz, ndrange=kernel_range)
    KernelAbstractions.synchronize(backend)
end

@kernel function remove_pressure_source_kernel!(cells, source_sign, ∇p, bx, by, bz) #Extend to 3D
    i = @index(Global)

    @uniform begin
        dpdx, dpdy, dpdz = ∇p.result.x, ∇p.result.y, ∇p.result.z
    end

    @inbounds begin
        (; volume) = cells[i]
        bx[i] -= source_sign * dpdx[i] * volume
        by[i] -= source_sign * dpdy[i] * volume
        bz[i] -= source_sign * dpdz[i] * volume
    end
end

# Pressure correction
function H!(Hv, U::VF, U_eqn, config) where {VF<:VectorField} # Extend to 3D!
    (; cells, cell_neighbours) = Hv.mesh
    (; hardware) = config
    (; backend, workgroup) = hardware

    A = _A(U_eqn)
    nzval, rowval, colptr = get_sparse_fields(A)
    (; bx, by, bz) = U_eqn.equation

    kernel_range = length(cells)
    kernel! = H_kernel!(backend, workgroup, kernel_range)
    kernel!(cells, cell_neighbours,
    nzval, colptr, rowval, bx, by, bz, U, Hv, ndrange=kernel_range)
    KernelAbstractions.synchronize(backend)
end

# Pressure correction kernel
@kernel function H_kernel!(cells::AbstractArray{Cell{TF,SV,UR}}, cell_neighbours,
    nzval, colptr, rowval, bx, by, bz, U, Hv) where {TF,SV,UR}
    i = @index(Global)

    @uniform begin
        Ux, Uy, Uz = U.x, U.y, U.z
        Hx, Hy, Hz = Hv.x, Hv.y, Hv.z
    end

    sumx = zero(TF)
    sumy = zero(TF)
    sumz = zero(TF)

    @inbounds begin
        (; faces_range) = cells[i]

        for ni ∈ faces_range
            nID = cell_neighbours[ni]
            zIndex = spindex(colptr, rowval, i, nID)
            val = nzval[zIndex]
            sumx += val * Ux[nID]
            sumy += val * Uy[nID]
            sumz += val * Uz[nID]
        end

        DIndex = spindex(colptr, rowval, i, i)
        D = nzval[DIndex]
        rD = 1/D
        Hx[i] = (bx[i] - sumx) * rD
        Hy[i] = (by[i] - sumy) * rD
        Hz[i] = (bz[i] - sumz) * rD
    end
end

## COURANT NUMBER

max_courant_number!(cellsCourant, model, config) = begin
    (; U) = model.momentum
    (; mesh) = U
    # (; cells) = mesh
    (; hardware, runtime) = config
    (; backend, workgroup) = hardware

    kernel_range = length(cellsCourant)
    kernel! = _max_courant_number!(backend, workgroup, kernel_range)
    kernel!(cellsCourant, U, runtime, mesh, ndrange=kernel_range)
    KernelAbstractions.synchronize(backend)
    return maximum(cellsCourant)
end

@kernel function _max_courant_number!(cellsCourant, U, runtime, mesh::Mesh3)
    i = @index(Global)
    @uniform cells = mesh.cells
    dt = runtime.dt
    # for i ∈ eachindex(U)
        umag = norm(U[i])
        volume = cells[i].volume
        # dx = sqrt(volume)
        dx = volume^0.333333
        cellsCourant[i] = umag*dt/dx
    # end
    # return co
end

@kernel function _max_courant_number!(cellsCourant, U, runtime, mesh::Mesh2)
    i = @index(Global)
    @uniform cells = mesh.cells
    dt = runtime.dt
    # for i ∈ eachindex(U)
        umag = norm(U[i])
        volume = cells[i].volume
        # dx = sqrt(volume)
        dx = volume^0.5
        cellsCourant[i] = umag*dt/dx
    # end
    # return co
end

courant_number(U, mesh::AbstractMesh, runtime) = begin
    F = _get_float(mesh)
    dt = runtime.dt
    co = zero(_get_float(mesh))
    cells = mesh.cells
    for i ∈ eachindex(U)
        umag = norm(U[i])
        volume = cells[i].volume
        dx = sqrt(volume)
        co = max(co, umag * dt / dx)
    end
    return co
end

@kernel function courant_number_kernel(U, cells, dt, F)
    i = @index(Global)
    co = zero(F)
    umag = norm(U[i])
    volume = cells[i].volume
    dx = sqrt(volume)
    co = max(co, umag * dt / dx)
end