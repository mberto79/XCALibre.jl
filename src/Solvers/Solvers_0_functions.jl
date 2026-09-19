export restart_fields!, restart_flux!
export flux!, update_nueff!, inverse_diagonal!, remove_pressure_source!, H!, correct_velocity!

## UPDATE EFFECTIVE VISCOSITY

function update_nueff!(nueff, nu, turb_model, config)
    (; mesh) = nueff
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(nueff)
    if typeof(turb_model) <: Laminar
        kernel! = _sized(update_nueff_laminar!, backend, workgroup, ndrange)
        kernel!(nu, nueff)
    else
        (; nutf) = turb_model
        kernel! = _sized(update_nueff_turbulent!, backend, workgroup, ndrange)
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

function flux!(phif::FS, psif::FV, config) where {FS<:FaceScalarField,FV<:FaceVectorField}
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(phif)
    kernel! = _sized(flux_kernel!, backend, workgroup, ndrange)
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

function flux!(phif::FS, psif::FV, rhof::FS, config) where {FS<:FaceScalarField,FV<:FaceVectorField}
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(phif)
    kernel! = _sized(_flux!, backend, workgroup, ndrange)
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

# halo=false leaves ghosts stale for a caller that exchanges rD together with Hv
function inverse_diagonal!(rD::S, eqn, config; halo=true) where {S<:ScalarField}
    (; hardware) = config
    (; backend, workgroup) = hardware
    A = eqn.equation.A # Or should I use A0
    nzval, colval, rowptr = get_sparse_fields(A)

    ndrange = length(rD)
    kernel! = _sized(_inverse_diagonal!, backend, workgroup, ndrange)
    kernel!(rD, nzval, colval, rowptr)
    # # KernelAbstractions.synchronize(backend)
    halo && sync!(rD, rD.mesh, config) # self-syncing seam (no-op serial)
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
    kernel! = _sized(_correct_velocity!, backend, workgroup, ndrange)
    kernel!(U, Hv, ∇p, rD)
    # no sync!: ghost U is already consistent (Hv/∇p/rD ghosts synced, kernel is pointwise)
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
    kernel! = _sized(_remove_pressure_source!, backend, workgroup, ndrange)
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
function H!(Hv, U::VF, U_eqn, config; halo=true) where {VF<:VectorField} # Extend to 3D!
    (; cells, cell_neighbours) = Hv.mesh
    (; hardware) = config
    (; backend, workgroup) = hardware

    A = _A(U_eqn)
    nzval, colval, rowptr = get_sparse_fields(A)
    (; bx, by, bz) = U_eqn.equation

    ndrange = length(cells)
    kernel! = _sized(_H!, backend, workgroup, ndrange)
    kernel!(cells, cell_neighbours,
        nzval, rowptr, colval, bx, by, bz, U, Hv)
    # # KernelAbstractions.synchronize(backend)
    halo && sync!(Hv, Hv.mesh, config) # self-syncing seam (no-op serial)
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

        rD = one(D)/D
        Hx[i] = (bx[i] - sumx)*rD
        Hy[i] = (by[i] - sumy)*rD
        Hz[i] = (bz[i] - sumz)*rD
    end
end

## COURANT NUMBER

# global_max seam (S5): serial = identity, Distribute = MPI.Allreduce(max). _base_mesh unwraps
# a DistributedMesh so the Mesh2/Mesh3 courant kernel still dispatches on the concrete geometry.
global_max(v, mesh) = v
_base_mesh(mesh) = mesh

max_courant_number!(cellsCourant, model, config) = begin
    (; U) = model.momentum
    (; mesh) = U
    # (; cells) = mesh
    (; hardware, runtime) = config
    (; backend, workgroup) = hardware

    ndrange = length(cellsCourant)
    kernel! = _sized(_max_courant_number!, backend, workgroup, ndrange)
    kernel!(cellsCourant, U, runtime, _base_mesh(mesh))
    # # KernelAbstractions.synchronize(backend)
    return global_max(maximum(cellsCourant), mesh)
end

@kernel function _max_courant_number!(cellsCourant, U, runtime, mesh::Mesh3)
    i = @index(Global)
    @uniform cells = mesh.cells
    dt = runtime.dt[1]
    umag = norm(U[i])
    volume = cells[i].volume
    dx = volume^(one(volume)/typeof(volume)(3))
    cellsCourant[i] = umag * dt / dx
end

@kernel function _max_courant_number!(cellsCourant, U, runtime, mesh::Mesh2)
    i = @index(Global)
    @uniform cells = mesh.cells
    dt = runtime.dt[1]
    umag = norm(U[i])
    volume = cells[i].volume
    dx = sqrt(volume)
    cellsCourant[i] = umag * dt / dx
end

## ALPHA COURANT NUMBER

max_alpha_courant_number!(cellsAlphaCourant, alpha, mdotf, model, config, dt) = begin
    (; U) = model.momentum
    (; mesh) = U
    (; hardware, runtime) = config
    (; backend, workgroup) = hardware

    ndrange = length(cellsAlphaCourant)
    kernel! = _sized(_max_alpha_courant_number!, backend, workgroup, ndrange)
    kernel!(cellsAlphaCourant, alpha, mdotf, runtime, dt, mesh)
    # # KernelAbstractions.synchronize(backend)
    return maximum(cellsAlphaCourant)
end


@kernel function _max_alpha_courant_number!(cellsAlphaCourant, alpha, mdotf, runtime, dt, mesh)
    i = @index(Global)

    @uniform cells = mesh.cells
    @uniform cell_faces = mesh.cell_faces

    # dt = runtime.dt
    volume = cells[i].volume
    alphaVal = alpha[i]

    nearInterfaceVal = nearInterface(alphaVal)
    sumAbsMdotf = zero(alphaVal)

    fr = cells[i].faces_range
    @inbounds for k in fr
        pointer = cell_faces[k]
        # faceID = mesh.faces[pointer]
        sumAbsMdotf += abs(mdotf[pointer])
    end

    cellsAlphaCourant[i] = dt * nearInterfaceVal * sumAbsMdotf / volume
end

@inline nearInterface(alpha) = ifelse((alpha > 0.01) & (alpha < 0.99), one(alpha), zero(alpha)) #Combines the two functions below into one
# @inline pos0(x) = ifelse(x >= zero(x), one(x), zero(x))
# @inline nearInterface(α) = pos0(α - 0.01) * pos0(0.99 - α)


update_dt!(runtime::Runtime{<:Any,<:Any,<:Any,Nothing}, ::Any) = nothing
update_dt!(runtime::Runtime{<:Any,<:Any,<:Any,Nothing}, ::Any, ::Any) = nothing

function update_dt!(runtime::Runtime{<:Any,<:Any,<:Any,<:AdaptiveTimeStepping}, courant)
    (; maxCo, maxGrow, minShrink) = runtime.adaptive

    courant_factor = maxCo / (courant + eps())
    new_dt_factor = clamp(courant_factor, minShrink, maxGrow)
    runtime.dt .= runtime.dt .* new_dt_factor
end

function update_dt!(runtime::Runtime{<:Any,<:Any,<:Any,<:AdaptiveTimeStepping}, courant, alphaCourant)
    (; maxCo, maxAlphaCo, maxGrow, minShrink) = runtime.adaptive

    courant_factor = maxCo / (courant + eps())
    alphaCourant_factor = maxAlphaCo / (alphaCourant + eps())
    
    new_dt_factor = min(courant_factor, alphaCourant_factor)
    new_dt_factor = clamp(new_dt_factor, minShrink, maxGrow)

    runtime.dt .= runtime.dt .* new_dt_factor
end

# NEW SECTION: restart hooks

# cell state and loop position before the initial calculations, face flux after them; the
# distributed module implements both for results written with output=OpenFOAM()
restart_fields!(mesh, model, ::Nothing, config) = (0, nothing)
restart_fields!(mesh, model, restart, config) =
    error("restart is supported on a distributed mesh whose results were written with output=OpenFOAM()")
restart_flux!(mesh, mdotf, ::Nothing, config) = nothing
restart_flux!(mesh, mdotf, restart, config) = restart_fields!(mesh, nothing, restart, config)
