export flux!, update_nueff!, inverse_diagonal!, remove_pressure_source!, H!, correct_velocity!

## UPDATE VISCOSITY

function update_nueff!(nueff, nu, turb_model, config)
    (; mesh) = nueff
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(nueff)
    if typeof(turb_model) <: Laminar
        kernel! = update_nueff_laminar!(_setup(backend, workgroup, ndrange)...)
        kernel!(nu, nueff)
    else
        (; nutf) = turb_model
        kernel! = update_nueff_turbulent!(_setup(backend, workgroup, ndrange)...)
        kernel!(nu, nutf, nueff)
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


## FLUX CALCULATION

# function flux!(phif::FS, psif::FV, config) where {FS<:FaceScalarField,FV<:FaceVectorField}
function flux!(phif::FS, psif::FV) where {FS<:FaceScalarField,FV<:FaceVectorField}
    # (; hardware) = config
    # (; backend, workgroup) = hardware
    (; backend, workgroup) = get_hardware(CONFIG)

    ndrange = length(phif)
    kernel! = flux_kernel!(_setup(backend, workgroup, ndrange)...)
    kernel!(phif, psif)
    # # KernelAbstractions.synchronize(backend)
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

# function flux!(phif::FS, psif::FV, rhof::FS, config) where {FS<:FaceScalarField,FV<:FaceVectorField}
function flux!(phif::FS, psif::FV, rhof::FS) where {FS<:FaceScalarField,FV<:FaceVectorField}
    # (; hardware) = config
    # (; backend, workgroup) = hardware
    (; backend, workgroup) = get_hardware(CONFIG)

    ndrange = length(phif)
    kernel! = _flux!(_setup(backend, workgroup, ndrange)...)
    kernel!(phif, psif, rhof)
    # # KernelAbstractions.synchronize(backend)
end

@kernel function _flux!(phif, psif, rhof)
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

get_hardware(config) = config[].hardware
get_boundaries(config) = config[].boundaries
get_configuration(config) = config

# function inverse_diagonal!(rD::S, eqn, config) where {S<:ScalarField}
function inverse_diagonal!(rD::S, eqn) where {S<:ScalarField}
    # (; hardware) = config
    # (; backend, workgroup) = hardware
    (; backend, workgroup) = get_hardware(CONFIG)
    A = eqn.equation.A # Or should I use A0
    nzval, colval, rowptr = get_sparse_fields(A)

    ndrange = length(rD)
    kernel! = _inverse_diagonal!(_setup(backend, workgroup, ndrange)...)
    kernel!(rD, nzval, colval, rowptr)
    # # KernelAbstractions.synchronize(backend)
end

@kernel function _inverse_diagonal!(rD, nzval, colval, rowptr)
    i = @index(Global)

    @uniform begin
        (; mesh, values) = rD
        cells = mesh.cells
    end

    @inbounds begin
        idx = spindex(rowptr, colval, i, i)
        D = nzval[idx]
        (; volume) = cells[i]
        values[i] = volume / D
    end
end

## VELOCITY CORRECTION

function correct_velocity!(U, Hv, ∇p, rD, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(U)
    kernel! = _correct_velocity!(_setup(backend, workgroup, ndrange)...)
    kernel!(U, Hv, ∇p, rD)
    # # KernelAbstractions.synchronize(backend)
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

    ndrange = length(bx)
    kernel! = _remove_pressure_source!(_setup(backend, workgroup, ndrange)...)
    kernel!(cells, source_sign, ∇p, bx, by, bz)
    # # KernelAbstractions.synchronize(backend)
end

@kernel function _remove_pressure_source!(cells, source_sign, ∇p, bx, by, bz) #Extend to 3D
    i = @index(Global)


    @inbounds begin
        (; volume) = cells[i]
        calc = source_sign*∇p[i]*volume
        bx[i] -= calc[1]
        by[i] -= calc[2]
        bz[i] -= calc[3]
    end
end

# Pressure correction
function H!(Hv, U::VF, U_eqn, config) where {VF<:VectorField} # Extend to 3D!
    (; cells, cell_neighbours) = Hv.mesh
    (; hardware) = config
    (; backend, workgroup) = hardware

    A = _A(U_eqn)
    nzval, colval, rowptr = get_sparse_fields(A)
    (; bx, by, bz) = U_eqn.equation

    ndrange = length(cells)
    kernel! = _H!(_setup(backend, workgroup, ndrange)...)
    kernel!(cells, cell_neighbours,
        nzval, rowptr, colval, bx, by, bz, U, Hv)
    # # KernelAbstractions.synchronize(backend)
end

# Pressure correction kernel
@kernel function _H!(cells::AbstractArray{Cell{TF,SV,UR}}, cell_neighbours,
    nzval, rowptr, colval, bx, by, bz, U, Hv) where {TF,SV,UR}
    i = @index(Global)

    @uniform begin
        Ux, Uy, Uz = U.x, U.y, U.z
        Hx, Hy, Hz = Hv.x, Hv.y, Hv.z
    end

    sumx = zero(TF)
    sumy = zero(TF)
    sumz = zero(TF)

    @inbounds begin
        # (; faces_range) = cells[i]

        # for ni ∈ faces_range
        #     nID = cell_neighbours[ni]
        #     zIndex = spindex(rowptr, colval, i, nID)
        #     val = nzval[zIndex]
        #     sumx += val * Ux[nID]
        #     sumy += val * Uy[nID]
        #     sumz += val * Uz[nID]
        # end

        start_index = rowptr[i]
        end_index = rowptr[i+1] - 1
        for nzi ∈ start_index:end_index
            nID = colval[nzi]
            val = nzval[nzi]
            sumx += val * Ux[nID]
            sumy += val * Uy[nID]
            sumz += val * Uz[nID]
        end

        DIndex = spindex(rowptr, colval, i, i)

        # remove diagonal contribution
        D = nzval[DIndex]
        sumx -= D*Ux[i]
        sumy -= D*Uy[i]
        sumz -= D*Uz[i]

        rD = 1.0/D
        Hx[i] = (bx[i] - sumx)*rD
        Hy[i] = (by[i] - sumy)*rD
        Hz[i] = (bz[i] - sumz)*rD
    end
end

## COURANT NUMBER

max_courant_number!(cellsCourant, model, config) = begin
    (; U) = model.momentum
    (; mesh) = U
    # (; cells) = mesh
    (; hardware, runtime) = config
    (; backend, workgroup) = hardware

    ndrange = length(cellsCourant)
    kernel! = _max_courant_number!(_setup(backend, workgroup, ndrange)...)
    kernel!(cellsCourant, U, runtime, mesh)
    # # KernelAbstractions.synchronize(backend)
    return maximum(cellsCourant)
end

@kernel function _max_courant_number!(cellsCourant, U, runtime, mesh::Mesh3)
    i = @index(Global)
    @uniform cells = mesh.cells
    dt = runtime.dt
    umag = norm(U[i])
    volume = cells[i].volume
    dx = volume^0.333333
    cellsCourant[i] = umag * dt / dx
end

@kernel function _max_courant_number!(cellsCourant, U, runtime, mesh::Mesh2)
    i = @index(Global)
    @uniform cells = mesh.cells
    dt = runtime.dt
    umag = norm(U[i])
    volume = cells[i].volume
    dx = volume^0.5
    cellsCourant[i] = umag * dt / dx
end