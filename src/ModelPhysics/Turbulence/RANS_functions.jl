# TO DO: These functions needs to be organised in a more sensible manner
function bound!(field, config)
    # Extract hardware configuration
    (; hardware) = config
    (; backend, workgroup) = hardware

    (; values, mesh) = field
    (; cells, cell_neighbours) = mesh

    # set up and launch kernel
    ndrange = length(values)
    kernel! = _bound!(_setup(backend, workgroup, ndrange)...)
    kernel!(values, cells, cell_neighbours)
    # KernelAbstractions.synchronize(backend)
end

@kernel function _bound!(values, cells, cell_neighbours)
    i = @index(Global)

    sum_flux = 0.0
    sum_area = 0
    average = 0.0
    @uniform mzero = eps(eltype(values)) # machine zero

    @inbounds begin
        for fi ∈ cells[i].faces_range
            cID = cell_neighbours[fi]
            sum_flux += max(values[cID], mzero) # bounded sum
            sum_area += 1
        end
        average = sum_flux/sum_area

        values[i] = max(
            max(
                values[i],
                average*signbit(values[i])
            ),
            mzero
        )
    end
end

y_plus_laminar(E, kappa) = begin
    yL = 11.0; for i ∈ 1:10; yL = log(max(yL*E, 1.0))/kappa; end
    yL
end

ω_vis(nu, y, beta1) = 6*nu/(beta1*y^2)

ω_log(k, y, cmu, kappa) = sqrt(k)/(cmu^0.25*kappa*y)

y_plus(k, nu, y, cmu) = cmu^0.25*y*sqrt(k)/nu

sngrad(Ui, Uw, delta, normal) = begin
    Udiff = (Ui - Uw)
    Up = Udiff - (Udiff⋅normal)*normal # parallel velocity difference
    grad = Up/delta
    return grad
end

mag(vector) = sqrt(vector[1]^2 + vector[2]^2 + vector[3]^2) 

nut_wall(nu, yplus, kappa, E::T) where T = begin
    max(nu*(yplus*kappa/log(max(E*yplus, 1.0 + 1e-4)) - 1.0), zero(T))
end

@generated correct_production!(P, fieldBCs, model, gradU, config) = begin
    BCs = fieldBCs.parameters
    any(BC -> BC <: KWallFunction, BCs) || return :(nothing)
    func_calls = Expr[]
    for i ∈ eachindex(BCs)
        call = quote
            accumulate_production!(
                weighted_production, wall_area, fieldBCs[$i], model, gradU, config,
            )
        end
        push!(func_calls, call)
    end
    quote
        (; backend, workgroup) = config.hardware
        n_cells = length(P)
        TF = _get_float(model.domain)
        weighted_production = KernelAbstractions.zeros(backend, TF, n_cells)
        wall_area = KernelAbstractions.zeros(backend, TF, n_cells)
        $(func_calls...)
        KernelAbstractions.synchronize(backend)

        kernel! = _apply_wall_average!(_setup(backend, workgroup, n_cells)...)
        kernel!(P, weighted_production, wall_area)
        KernelAbstractions.synchronize(backend)
        nothing
    end
end

accumulate_production!(weighted_production, wall_area, BC, model, gradU, config) = nothing

function accumulate_production!(
    weighted_production, wall_area, BC::KWallFunction, model, gradU, config,
)
    (; hardware) = config
    (; backend, workgroup) = hardware
    
    # Deconstruct mesh to required fields
    mesh = model.domain
    (; faces, boundary_cellsID) = mesh
    facesID_range = BC.IDs_range
    start_ID = facesID_range[1]

    ndrange = length(facesID_range)
    kernel! = _accumulate_production!(_setup(backend, workgroup, ndrange)...)
    kernel!(
        weighted_production,
        wall_area,
        BC,
        model.fluid,
        model.momentum,
        model.turbulence,
        faces,
        boundary_cellsID,
        start_ID,
        gradU,
    )
end

@kernel function _accumulate_production!(
    weighted_production,
    wall_area,
    BC::KWallFunction,
    fluid,
    momentum,
    turbulence,
    faces,
    boundary_cellsID,
    start_ID,
    gradU,
)
    i = @index(Global)
    fID = i + start_ID - 1

    @inbounds begin
        (; kappa, cmu, E, yPlusLam) = BC.value
        (; nu) = fluid
        (; U, Uf) = momentum
        (; k) = turbulence

        cID = boundary_cellsID[fID]
        face = faces[fID]
        (; area, delta, normal) = face
        nuc = nu[cID]
        u_star = cmu^oftype(cmu, 0.25)*sqrt(k[cID])
        dUdy = u_star/(kappa*delta)
        yplus = y_plus(k[cID], nuc, delta, cmu)
        nutw = nut_wall(nuc, yplus, kappa, E)
        Uw = Uf[fID]
        mag_grad_U = mag(sngrad(U[cID], Uw, delta, normal))
        production = ifelse(
            yplus > yPlusLam,
            (nuc + nutw)*mag_grad_U*dUdy,
            zero(nuc),
        )
        Atomix.@atomic weighted_production[cID] += area*production
        Atomix.@atomic wall_area[cID] += area
    end
end

@kernel function _apply_wall_average!(field, weighted_values, wall_area)
    cID = @index(Global)
    @inbounds begin
        area = wall_area[cID]
        if area > zero(area)
            field[cID] = weighted_values[cID]/area
        end
    end
end

@generated function correct_eddy_viscosity!(νtf, nutBCs, model, config)
    unpacked_BCs = []
    for i ∈ 1:length(nutBCs.parameters)
        unpack = quote
            correct_nut_wall!(νtf, nutBCs[$i], model, config)
        end
        push!(unpacked_BCs, unpack)
    end
    quote
    $(unpacked_BCs...) 
    end
end

correct_nut_wall!(nutf, BC, model, config) = nothing

function correct_nut_wall!(νtf, BC::NutWallFunction, model, config)
    # backend = _get_backend(mesh)
    (; hardware) = config
    (; backend, workgroup) = hardware
    
    # Deconstruct mesh to required fields
    mesh = model.domain
    (; faces, boundary_cellsID, boundaries) = mesh

    # Extract physics models
    (; fluid, turbulence) = model

    # facesID_range = get_boundaries(BC, boundaries)
    # boundaries_cpu = get_boundaries(boundaries)
    # facesID_range = boundaries_cpu[BC.ID].IDs_range
    facesID_range = BC.IDs_range
    start_ID = facesID_range[1]

    # Execute apply boundary conditions kernel
    ndrange=length(facesID_range)
    kernel! = _correct_nut_wall!(_setup(backend, workgroup, ndrange)...)
    kernel!(νtf.values, fluid, turbulence, BC, faces, boundary_cellsID, start_ID)
end

@kernel function _correct_nut_wall!(
    values, fluid, turbulence, BC::NutWallFunction, faces, boundary_cellsID, start_ID)
    i = @index(Global)
    fID = i + start_ID - 1 # Redefine thread index to become face ID

    (; kappa, beta1, cmu, B, E, yPlusLam) = BC.value
    (; nu) = fluid
    (; k) = turbulence
    
    cID = boundary_cellsID[fID]
    face = faces[fID]
    # nuf = nu[fID]
    (; delta)= face
    # yplus = y_plus(k[cID], nuf, delta, cmu)
    nuc = nu[cID]
    yplus = y_plus(k[cID], nuc, delta, cmu)
    nutw = nut_wall(nuc, yplus, kappa, E)
    if yplus > yPlusLam
        values[fID] = nutw
    else
        values[fID] = 0.0
    end
end

function correct_nut_wall!(νtf, BC::NutMixingLengthWallFunction, model, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    mesh = model.domain
    (; faces, boundary_cellsID) = mesh
    (; fluid, momentum, turbulence) = model

    facesID_range = BC.IDs_range
    start_ID = facesID_range[1]

    ndrange = length(facesID_range)
    kernel! = _correct_nut_wall_mixing_length!(_setup(backend, workgroup, ndrange)...)
    kernel!(νtf.values, fluid, momentum, turbulence, BC, faces, boundary_cellsID, start_ID)
end

@kernel function _correct_nut_wall_mixing_length!(
    values, fluid, momentum, turbulence, BC::NutMixingLengthWallFunction, faces, boundary_cellsID, start_ID)
    i = @index(Global)
    fID = i + start_ID - 1

    (; kappa, E, yPlusLam) = BC.value
    (; nu) = fluid
    (; U) = momentum
    (; nut) = turbulence

    cID = boundary_cellsID[fID]
    face = faces[fID]
    (; delta, normal) = face
    nuc = nu[cID]

    # Tangential velocity magnitude at cell centre (wall velocity = 0)
    Ucell = U[cID]
    U_tang = Ucell - (Ucell ⋅ normal) * normal
    U_tang_mag = mag(U_tang)

    # Newton iteration: solve U_tang_mag = (u_tau/kappa)*ln(E*u_tau*delta/nu) for u_tau
    # Initial guess from viscous sublayer: u_tau ≈ sqrt(nu*|U_t|/delta)
    u_tau = sqrt(nuc * U_tang_mag / delta + eltype(values)(1e-20))
    for _ in 1:10
        yp  = u_tau * delta / nuc
        lv  = log(max(E * yp, eltype(values)(1.0 + 1e-4)))
        f   = U_tang_mag * kappa - u_tau * lv
        df  = -(lv + one(eltype(values)))
        u_tau = max(u_tau - f / df, eltype(values)(1e-20))
    end

    yplus = u_tau * delta / nuc
    nutw  = nut_wall(nuc, yplus, kappa, E)

    if yplus > yPlusLam
        values[fID] = nutw
        nut[cID] = nutw
    else
        values[fID] = zero(eltype(values))
    end
end

@generated constrain_equation!(eqn, fieldBCs, model, config) = begin
    BCs = fieldBCs.parameters
    any(BC -> BC <: OmegaWallFunction, BCs) || return :(nothing)
    func_calls = Expr[]
    for i ∈ eachindex(BCs)
        call = quote
            accumulate_omega_constraint!(
                weighted_omega, wall_area, fieldBCs[$i], model, config,
            )
        end
        push!(func_calls, call)
    end
    quote
        (; backend, workgroup) = config.hardware
        A = _A(eqn)
        b = _b(eqn, nothing)
        colval = _colval(A)
        rowptr = _rowptr(A)
        nzval = _nzval(A)
        n_cells = length(b)
        TF = eltype(nzval)
        weighted_omega = KernelAbstractions.zeros(backend, TF, n_cells)
        wall_area = KernelAbstractions.zeros(backend, TF, n_cells)
        $(func_calls...)
        KernelAbstractions.synchronize(backend)

        kernel! = _apply_omega_constraints!(_setup(backend, workgroup, n_cells)...)
        kernel!(weighted_omega, wall_area, colval, rowptr, nzval, b)
        KernelAbstractions.synchronize(backend)
        nothing
    end
end

accumulate_omega_constraint!(weighted_omega, wall_area, BC, model, config) = nothing

function accumulate_omega_constraint!(
    weighted_omega, wall_area, BC::OmegaWallFunction, model, config,
)
    (; hardware) = config
    (; backend, workgroup) = hardware
    mesh = model.domain
    (; faces, boundary_cellsID) = mesh
    facesID_range = BC.IDs_range
    start_ID = facesID_range[1]

    ndrange = length(facesID_range)
    kernel! = _accumulate_omega_constraint!(_setup(backend, workgroup, ndrange)...)
    kernel!(
        weighted_omega,
        wall_area,
        model.turbulence,
        model.fluid,
        BC,
        faces,
        start_ID,
        boundary_cellsID,
    )
end

@kernel function _accumulate_omega_constraint!(
    weighted_omega,
    wall_area,
    turbulence,
    fluid,
    BC::OmegaWallFunction,
    faces,
    start_ID,
    boundary_cellsID,
)
    i = @index(Global)
    fID = i + start_ID - 1

    @uniform begin
        nu = fluid.nu
        k = turbulence.k
        (; kappa, beta1, cmu, B, E, yPlusLam) = BC.value
    end
    @inbounds begin
        cID = boundary_cellsID[fID]
        face = faces[fID]
        (; area) = face
        y = face.delta
        ωvis = ω_vis(nu[cID], y, beta1)
        ωlog = ω_log(k[cID], y, cmu, kappa)
        yplus = y_plus(k[cID], nu[cID], y, cmu)
        ωc = ifelse(yplus > yPlusLam, ωlog, ωvis)
        Atomix.@atomic weighted_omega[cID] += area*ωc
        Atomix.@atomic wall_area[cID] += area
    end
end

@kernel function _apply_omega_constraints!(
    weighted_omega, wall_area, colval, rowptr, nzval, b,
)
    cID = @index(Global)
    @inbounds begin
        area = wall_area[cID]
        if area > zero(area)
            ωc = weighted_omega[cID]/area
            z = zero(eltype(nzval))
            for nzi ∈ rowptr[cID]:(rowptr[cID+1] - 1)
                nzval[nzi] = z
            end
            cIndex = spindex(rowptr, colval, cID, cID)
            nzval[cIndex] = one(eltype(nzval))
            b[cID] = ωc
        end
    end
end

# @generated constrain_boundary!(field, fieldBCs, model, config) = begin
#     BCs = fieldBCs.parameters
#     func_calls = Expr[]
#     for i ∈ eachindex(BCs)
#         call = quote
#             set_cell_value!(field, fieldBCs[$i], model, config)
#         end
#         push!(func_calls, call)
#     end
#     quote
#     $(func_calls...)
#     nothing
#     end 
# end

# set_cell_value!(field, BC, model, config) = nothing

# function set_cell_value!(field, BC::OmegaWallFunction, model, config)
#     # backend = _get_backend(mesh)
#     (; hardware) = config
#     (; backend, workgroup) = hardware
    
#     # Deconstruct mesh to required fields
#     mesh = model.domain
#     (; faces, boundaries, boundary_cellsID) = mesh
#     (; fluid, turbulence) = model
#     # turbFields = turbulence.fields

#     # facesID_range = get_boundaries(BC, boundaries)
#     boundaries_cpu = get_boundaries(boundaries)
#     facesID_range = boundaries_cpu[BC.ID].IDs_range
#     start_ID = facesID_range[1]

#     # Execute apply boundary conditions kernel
        # ndrange=length(facesID_range)
#     kernel! = _set_cell_value!(_setup(backend, workgroup, ndrange)...)
#     kernel!(
#         field, turbulence, fluid, BC, faces, start_ID, boundary_cellsID
#     )
# end

# @kernel function _set_cell_value!(field, turbulence, fluid, BC, faces, start_ID, boundary_cellsID)
#     i = @index(Global)
#     fID = i + start_ID - 1 # Redefine thread index to become face ID

#     @uniform begin
#         (; nu) = fluid
#         (; k) = turbulence
#         (; kappa, beta1, cmu, B, E, yPlusLam) = BC.value
#         (; values) = field
#         ωc = zero(eltype(values))
#     end


#     @inbounds begin
#         cID = boundary_cellsID[fID]
#         face = faces[fID]
#         y = face.delta
#         ωvis = ω_vis(nu[cID], y, beta1)
#         ωlog = ω_log(k[cID], y, cmu, kappa)
#         yplus = y_plus(k[cID], nu[cID], y, cmu) 

#         if yplus > yPlusLam 
#             ωc = ωlog
#         else
#             ωc = ωvis
#         end

#         values[cID] = ωc # needs to be atomic?
#     end
# end
