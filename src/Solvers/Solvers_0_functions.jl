export flux!, update_nueff!, residual!, inverse_diagonal!, remove_pressure_source!, H!, correct_velocity!

## UPDATE VISCOSITY

function update_nueff!(nueff, nu, turb_model, config)
    (; mesh) = nueff
    # backend = _get_backend(mesh)
    (; hardware) = config
    (; backend, workgroup) = hardware

    if turb_model === nothing
        kernel! = update_nueff_laminar!(backend, workgroup)
        kernel!(nu, nueff, ndrange=length(nueff))
    else
        (; νtf) = turb_model
        kernel! = update_nueff_turbulent!(backend, workgroup)
        kernel!(nu, νtf, nueff, ndrange=length(nueff))
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

function residual!(Residual, equation, phi, iteration, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    (; A, b, R, Fx) = equation
    values = phi.values

    # backend = _get_backend(phi.mesh)

    # sparse_matmul!(A, values, Fx, config)
    Fx .= A * values
    # KernelAbstractions.synchronize(backend)

    @inbounds @. R = abs(Fx - b)^2
    # KernelAbstractions.synchronize(backend)

    # res = sqrt(mean(R))/norm(b)
    Residual[iteration] = sqrt(mean(R)) / norm(b)
    nothing
end

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

    kernel! = matmul_copy_zeros_kernel!(backend, workgroup)
    kernel!(c, fzero, ndrange=length(c))
    KernelAbstractions.synchronize(backend)

    kernel! = sparse_matmul_kernel!(backend, workgroup)
    kernel!(nzval_array, rowval_array, colptr_array, b, c, ndrange=length(c))
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

    kernel! = flux_kernel!(backend, workgroup)
    kernel!(phif, psif, ndrange=length(phif))
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

volumes(mesh) = [mesh.cells[i].volume for i ∈ eachindex(mesh.cells)]

# INVERSE DIAGONAL CALCULATION

function inverse_diagonal!(rD::S, eqn, config) where {S<:ScalarField}
    (; hardware) = config
    (; backend, workgroup) = hardware
    A = eqn.equation.A
    nzval, rowval, colptr = get_sparse_fields(A)

    kernel! = inverse_diagonal_kernel!(backend, workgroup)
    kernel!(rD, nzval, rowval, colptr, ndrange=length(rD))
    KernelAbstractions.synchronize(backend)
end

# @kernel function inverse_diagonal_kernel!(ione, colptr, rowval, nzval, cells, values)
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
    Ux = U.x
    Uy = U.y
    Uz = U.z
    Hvx = Hv.x
    Hvy = Hv.y
    Hvz = Hv.z
    dpdx = ∇p.result.x
    dpdy = ∇p.result.y
    dpdz = ∇p.result.z
    rDvalues = rD.values
    # backend = _get_backend(U.mesh)
    (; hardware) = config
    (; backend, workgroup) = hardware

    kernel! = correct_velocity_kernel!(backend, workgroup)
    kernel!(rDvalues, Ux, Hvx, dpdx, Uy, Hvy, dpdy, Uz, Hvz, dpdz, ndrange=length(Ux))
    KernelAbstractions.synchronize(backend)
end

@kernel function correct_velocity_kernel!(rDvalues,
    Ux, Hvx, dpdx,
    Uy, Hvy, dpdy,
    Uz, Hvz, dpdz)
    i = @index(Global)

    @inbounds begin
        rDvalues_i = rDvalues[i]
        Ux[i] = Hvx[i] - dpdx[i] * rDvalues_i
        Uy[i] = Hvy[i] - dpdy[i] * rDvalues_i
        Uz[i] = Hvz[i] - dpdz[i] * rDvalues_i
    end
end

## PRESSURE CORRECTION AND SOURCE REMOVAL

remove_pressure_source!(ux_eqn::M1, uy_eqn::M2, uz_eqn::M3, ∇p, config) where {M1,M2,M3} = begin # Extend to 3D
    # backend = _get_backend(get_phi(ux_eqn).mesh)
    (; hardware) = config
    (; backend, workgroup) = hardware
    cells = get_phi(ux_eqn).mesh.cells
    source_sign = get_source_sign(ux_eqn, 1)
    dpdx, dpdy, dpdz = ∇p.result.x, ∇p.result.y, ∇p.result.z
    bx, by, bz = ux_eqn.equation.b, uy_eqn.equation.b, uz_eqn.equation.b

    kernel! = remove_pressure_source_kernel!(backend, workgroup)
    kernel!(cells, source_sign, dpdx, dpdy, dpdz, bx, by, bz, ndrange=length(bx))
    KernelAbstractions.synchronize(backend)
end

@kernel function remove_pressure_source_kernel!(cells, source_sign, dpdx, dpdy, dpdz, bx, by, bz) #Extend to 3D
    i = @index(Global)

    @inbounds begin
        (; volume) = cells[i]
        Atomix.@atomic bx[i] -= source_sign * dpdx[i] * volume
        Atomix.@atomic by[i] -= source_sign * dpdy[i] * volume
        Atomix.@atomic bz[i] -= source_sign * dpdz[i] * volume
    end
end

# Pressure correction
function H!(Hv, v::VF, ux_eqn, uy_eqn, uz_eqn, config) where {VF<:VectorField} # Extend to 3D!
    (; x, y, z, mesh) = Hv
    (; cells, faces) = mesh
    (; cells, cell_neighbours, faces) = mesh
    # backend = _get_backend(mesh)
    (; hardware) = config
    (; backend, workgroup) = hardware

    Ax = _A(ux_eqn)
    bx = _b(ux_eqn)
    nzval_x, rowval_x, colptr_x = get_sparse_fields(Ax)

    Ay = _A(uy_eqn)
    by = _b(uy_eqn)
    nzval_y, rowval_y, colptr_y = get_sparse_fields(Ay)

    Az = _A(uz_eqn)
    bz = _b(uz_eqn)
    nzval_z, rowval_z, colptr_z = get_sparse_fields(Az)

    vx, vy, vz = v.x, v.y, v.z

    kernel! = H_kernel!(backend, workgroup)
    kernel!(cells, cell_neighbours,
        nzval_x, colptr_x, rowval_x, bx, vx,
        nzval_y, colptr_y, rowval_y, by, vy,
        nzval_z, colptr_z, rowval_z, bz, vz,
        x, y, z, ndrange=length(cells))
    KernelAbstractions.synchronize(backend)
end

# Pressure correction kernel
@kernel function H_kernel!(cells::AbstractArray{Cell{TF,SV,UR}}, cell_neighbours,
    nzval_x, colptr_x, rowval_x, bx, vx,
    nzval_y, colptr_y, rowval_y, by, vy,
    nzval_z, colptr_z, rowval_z, bz, vz,
    x, y, z) where {TF,SV,UR}
    i = @index(Global)
    sumx = zero(TF)
    sumy = zero(TF)
    sumz = zero(TF)

    @inbounds begin
        (; faces_range) = cells[i]

        for ni ∈ faces_range
            nID = cell_neighbours[ni]
            xIndex = spindex(colptr_x, rowval_x, i, nID)
            yIndex = spindex(colptr_y, rowval_y, i, nID)
            zIndex = spindex(colptr_z, rowval_z, i, nID)
            sumx += nzval_x[xIndex] * vx[nID]
            sumy += nzval_y[yIndex] * vy[nID]
            sumz += nzval_z[zIndex] * vz[nID]
        end

        # D = view(Ax, i, i)[1] # add check to use max of Ax or Ay)
        DIndex = spindex(colptr_x, rowval_x, i, i)
        Dx = nzval_x[DIndex]
        Dy = nzval_y[DIndex]
        Dz = nzval_z[DIndex]
        Dmax = max(Dx, Dy, Dz)
        rD = 1 / Dmax
        x[i] = (bx[i] - sumx) * rD
        y[i] = (by[i] - sumy) * rD
        z[i] = (bz[i] - sumz) * rD
    end
end

## COURANT NUMBER

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