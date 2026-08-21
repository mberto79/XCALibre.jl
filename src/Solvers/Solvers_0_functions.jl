export flux!, update_nueff!, inverse_diagonal!, remove_pressure_source!, H!, correct_velocity!

## UPDATE EFFECTIVE VISCOSITY

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

function flux!(phif::FS, psif::FV, config) where {FS<:FaceScalarField,FV<:FaceVectorField}
    (; hardware) = config
    (; backend, workgroup) = hardware

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

function flux!(phif::FS, psif::FV, rhof::FS, config) where {FS<:FaceScalarField,FV<:FaceVectorField}
    (; hardware) = config
    (; backend, workgroup) = hardware

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

function inverse_diagonal!(rD::S, eqn, config) where {S<:ScalarField}
    (; hardware) = config
    (; backend, workgroup) = hardware
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

        rD = one(D)/D
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

# DIRECTIONAL, not isotropic.
#
# This previously used `dx = volume^(1/3)` (or `sqrt(volume)` in 2D) - a single
# isotropic length for the whole cell. That is only the cell's size when the cell
# is roughly cubic, and it silently flatters anisotropic meshes: on the LH2 pipe's
# near-wall cells (aspect ratio ~44:1, volume 5.01e-11 m^3, total face area
# 1.43e-06 m^2) it gives `dx = 369 um` against a true wall-normal spacing of
# ~35 um, so the reported Courant number was 10.6x too SMALL.
#
# That matters twice over. Adaptive stepping keyed on `maxCo` was letting the step
# grow ~10x beyond what the mesh supports, and the ordinary Courant number
# disagreed with the ALPHA Courant number - which is already flux-based - by the
# same factor, making the pair impossible to interpret together.
#
# The standard definition instead sums the flux through the faces,
#
#     Co = 0.5*dt*sum_f |U . n_f| A_f / V
#
# which is exact for uniform flow through a cuboid (only the two faces normal to
# the flow contribute, giving `u*dt/h` on the spacing in the FLOW direction) and
# degrades gracefully on skewed cells. Same convention as the alpha Courant, so
# the two are now directly comparable.
@kernel function _max_courant_number!(cellsCourant, U, runtime, mesh::Mesh3)
    i = @index(Global)
    @uniform cells = mesh.cells
    @uniform cell_faces = mesh.cell_faces
    @uniform faces = mesh.faces
    dt = runtime.dt[1]
    Ui = U[i]
    cell = cells[i]
    volume = cell.volume
    flux = zero(volume)
    @inbounds for k in cell.faces_range
        face = faces[cell_faces[k]]
        n = face.normal
        flux += abs(Ui[1]*n[1] + Ui[2]*n[2] + Ui[3]*n[3])*face.area
    end
    cellsCourant[i] = 0.5 * dt * flux / volume
end

@kernel function _max_courant_number!(cellsCourant, U, runtime, mesh::Mesh2)
    i = @index(Global)
    @uniform cells = mesh.cells
    @uniform cell_faces = mesh.cell_faces
    @uniform faces = mesh.faces
    dt = runtime.dt[1]
    Ui = U[i]
    cell = cells[i]
    volume = cell.volume
    flux = zero(volume)
    @inbounds for k in cell.faces_range
        face = faces[cell_faces[k]]
        n = face.normal
        flux += abs(Ui[1]*n[1] + Ui[2]*n[2] + Ui[3]*n[3])*face.area
    end
    cellsCourant[i] = 0.5 * dt * flux / volume
end

## ALPHA COURANT NUMBER

max_alpha_courant_number!(cellsAlphaCourant, alpha, mdotf, model, config, dt) = begin
    (; U) = model.momentum
    (; mesh) = U
    (; hardware, runtime) = config
    (; backend, workgroup) = hardware

    ndrange = length(cellsAlphaCourant)
    kernel! = _max_alpha_courant_number!(_setup(backend, workgroup, ndrange)...)
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