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

# A wall cell can own several faces of the same patch and faces on several patches.
# Wall functions therefore sum over the contributing faces and divide by their number, so
# each face carries equal weight. Assigning the cell value directly instead made the result
# depend on which face won the race, both within a patch and across patches.
wall_cell_accumulators(mesh, config) = begin
    (; backend) = config.hardware
    n = length(mesh.cells)
    TF = _get_float(mesh)
    KernelAbstractions.zeros(backend, TF, n), KernelAbstractions.zeros(backend, TF, n)
end

# Every patch must be summed before any cell is averaged, so the two passes each run
# over all patches. The averaging write is the same from every face of a cell, which
# keeps it free of the race it replaces.
average_wall_cells!(values, BC, sums, counts, model, config) = nothing

function average_wall_cells!(
    values, BC::Union{KWallFunction,OmegaWallFunction,NutMixingLengthWallFunction},
    sums, counts, model, config)
    (; backend, workgroup) = config.hardware
    boundary_cellsID = model.domain.boundary_cellsID
    ndrange = length(BC.IDs_range)
    kernel! = _average_wall_cells!(backend)
    kernel!(values, sums, counts, boundary_cellsID, BC.IDs_range[1];
        _dynamic_setup(backend, workgroup, ndrange)...)
end

@kernel function _average_wall_cells!(values, sums, counts, boundary_cellsID, start_ID)
    i = @index(Global)
    fID = i + start_ID - 1
    @inbounds begin
        cID = boundary_cellsID[fID]
        count = counts[cID]
        # A cell with no contribution keeps the value the model gave it.
        count > zero(count) && (values[cID] = sums[cID]/count)
    end
end

@generated correct_production!(P, fieldBCs, model, gradU, config) = begin
    BCs = fieldBCs.parameters
    any(BC -> BC <: KWallFunction, BCs) || return :(nothing)
    sum_calls = [:(set_production!(sums, counts, fieldBCs[$i], model, gradU, config)) for i ∈ eachindex(BCs)]
    avg_calls = [:(average_wall_cells!(P.values, fieldBCs[$i], sums, counts, model, config)) for i ∈ eachindex(BCs)]
    quote
        sums, counts = wall_cell_accumulators(model.domain, config)
        $(sum_calls...)
        $(avg_calls...)
        nothing
    end
end

set_production!(sums, counts, BC, model, gradU, config) = nothing

function set_production!(sums, counts, BC::KWallFunction, model, gradU, config)
    # backend = _get_backend(mesh)
    (; hardware) = config
    (; backend, workgroup) = hardware
    
    # Deconstruct mesh to required fields
    mesh = model.domain
    (; faces, boundary_cellsID, boundaries) = mesh

    # Extract physics models
    (; fluid, momentum, turbulence) = model

    # facesID_range = get_boundaries(BC, boundaries)
    # boundaries_cpu = get_boundaries(boundaries)
    # facesID_range = boundaries_cpu[BC.ID].IDs_range
    facesID_range = BC.IDs_range
    start_ID = facesID_range[1]

    # Execute apply boundary conditions kernel
    ndrange = length(facesID_range)
    kernel! = _set_production!(backend)
    kernel!(
        sums, counts, BC, fluid, momentum, turbulence, faces, boundary_cellsID,
        start_ID, gradU;
        _dynamic_setup(backend, workgroup, ndrange)...
    )
end

@kernel function _set_production!(
    sums, counts, BC::KWallFunction, fluid, momentum, turbulence, faces,
    boundary_cellsID, start_ID, gradU)
    i = @index(Global)
    fID = i + start_ID - 1 # Redefine thread index to become face ID

    (; kappa, beta1, cmu, B, E, yPlusLam) = BC.value
    (; nu, rho) = fluid
    (; U, Uf) = momentum
    (; k, nut) = turbulence

    cID = boundary_cellsID[fID]
    face = faces[fID]
    nuc = nu[cID]
    (; delta, normal)= face
    uStar = cmu^0.25*sqrt(k[cID])
    dUdy = uStar/(kappa*delta)
    yplus = y_plus(k[cID], nuc, delta, cmu)
    nutw = nut_wall(nuc, yplus, kappa, E)
    Uw = Uf[fID]
    mag_grad_U = mag(sngrad(U[cID], Uw, delta, normal))
    Pf = yplus > yPlusLam ? rho[cID]*(nu[cID] + nutw)*mag_grad_U*dUdy : zero(eltype(sums))
    Atomix.@atomic sums[cID] += Pf
    Atomix.@atomic counts[cID] += one(eltype(counts))
end

# Only the mixing-length variant writes a cell value, so the averaging phases are
# emitted only when one is present.
@generated function correct_eddy_viscosity!(νtf, nutBCs, model, config)
    BCs = nutBCs.parameters
    calls = [:(correct_nut_wall!(νtf, nutBCs[$i], sums, counts, model, config)) for i ∈ eachindex(BCs)]
    any(BC -> BC <: NutMixingLengthWallFunction, BCs) || return quote
        sums = counts = nothing
        $(calls...)
        nothing
    end
    avg_calls = [:(average_wall_cells!(model.turbulence.nut.values, nutBCs[$i], sums, counts, model, config)) for i ∈ eachindex(BCs)]
    quote
        sums, counts = wall_cell_accumulators(model.domain, config)
        $(calls...)
        $(avg_calls...)
        nothing
    end
end

correct_nut_wall!(nutf, BC, sums, counts, model, config) = nothing

function correct_nut_wall!(νtf, BC::NutWallFunction, sums, counts, model, config)
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
    kernel! = _correct_nut_wall!(backend)
    kernel!(νtf.values, fluid, turbulence, BC, faces, boundary_cellsID, start_ID;
        _dynamic_setup(backend, workgroup, ndrange)...)
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

function correct_nut_wall!(νtf, BC::NutMixingLengthWallFunction, sums, counts, model, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    mesh = model.domain
    (; faces, boundary_cellsID) = mesh
    (; fluid, momentum, turbulence) = model

    facesID_range = BC.IDs_range
    start_ID = facesID_range[1]

    ndrange = length(facesID_range)
    kernel! = _correct_nut_wall_mixing_length!(backend)
    kernel!(νtf.values, sums, counts, fluid, momentum, turbulence, BC, faces,
        boundary_cellsID, start_ID; _dynamic_setup(backend, workgroup, ndrange)...)
end

@kernel function _correct_nut_wall_mixing_length!(
    values, sums, counts, fluid, momentum, turbulence, BC::NutMixingLengthWallFunction, faces, boundary_cellsID, start_ID)
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
        Atomix.@atomic sums[cID] += nutw
        Atomix.@atomic counts[cID] += one(eltype(counts))
    else
        values[fID] = zero(eltype(values))
    end
end

@generated constrain_equation!(eqn, fieldBCs, model, config) = begin
    BCs = fieldBCs.parameters
    any(BC -> BC <: OmegaWallFunction, BCs) || return :(nothing)
    fix_calls = [:(fix_wall_row!(eqn, fieldBCs[$i], model, config)) for i ∈ eachindex(BCs)]
    constrain_calls = [:(constrain!(sums, counts, fieldBCs[$i], model, config)) for i ∈ eachindex(BCs)]
    avg_calls = [:(average_wall_cells!(_b(eqn, nothing), fieldBCs[$i], sums, counts, model, config)) for i ∈ eachindex(BCs)]
    quote
        sums, counts = wall_cell_accumulators(model.domain, config)
        $(fix_calls...)
        $(constrain_calls...)
        $(avg_calls...)
        nothing
    end
end

fix_wall_row!(eqn, BC, model, config) = nothing

# Pins the cell to the wall value: every contributing face writes the same row, so the
# repeated writes are harmless, and the source is then averaged over those faces.
function fix_wall_row!(eqn, BC::OmegaWallFunction, model, config)
    (; backend, workgroup) = config.hardware
    A = _A(eqn)
    b = _b(eqn, nothing)
    boundary_cellsID = model.domain.boundary_cellsID
    ndrange = length(BC.IDs_range)
    kernel! = _fix_wall_row!(backend)
    kernel!(_colval(A), _rowptr(A), _nzval(A), b, boundary_cellsID, BC.IDs_range[1];
        _dynamic_setup(backend, workgroup, ndrange)...)
end

@kernel function _fix_wall_row!(colval, rowptr, nzval, b, boundary_cellsID, start_ID)
    i = @index(Global)
    fID = i + start_ID - 1
    @inbounds begin
        cID = boundary_cellsID[fID]
        z = zero(eltype(nzval))
        for nzi ∈ rowptr[cID]:(rowptr[cID+1] - 1)
            nzval[nzi] = z
        end
        nzval[spindex(rowptr, colval, cID, cID)] = one(eltype(nzval))
        b[cID] = z
    end
end

constrain!(sums, counts, BC, model, config) = nothing

function constrain!(sums, counts, BC::OmegaWallFunction, model, config)

    # backend = _get_backend(mesh)
    (; hardware) = config
    (; backend, workgroup) = hardware

    # Deconstruct mesh to required fields
    mesh = model.domain
    (; faces, boundaries, boundary_cellsID) = mesh

    fluid = model.fluid 
    # turbFields = model.turbulence.fields
    turbulence = model.turbulence

    # facesID_range = get_boundaries(BC, boundaries)
    # boundaries_cpu = get_boundaries(boundaries)
    # facesID_range = boundaries_cpu[BC.ID].IDs_range
    facesID_range = BC.IDs_range
    start_ID = facesID_range[1]

    # Execute apply boundary conditions kernel
    ndrange = length(facesID_range)
    kernel! = _constrain!(backend)
    kernel!(
        turbulence, fluid, BC, faces, start_ID, boundary_cellsID, sums, counts;
        _dynamic_setup(backend, workgroup, ndrange)...
    )
end

@kernel function _constrain!(turbulence, fluid, BC::OmegaWallFunction, faces, start_ID, boundary_cellsID, sums, counts)
    i = @index(Global)
    fID = i + start_ID - 1 # Redefine thread index to become face ID

    @uniform begin
        nu = fluid.nu
        k = turbulence.k
        (; kappa, beta1, cmu, B, E, yPlusLam) = BC.value
    end
    ωc = zero(eltype(sums))

    @inbounds begin
        cID = boundary_cellsID[fID]
        face = faces[fID]
        y = face.delta
        ωvis = ω_vis(nu[cID], y, beta1)
        ωlog = ω_log(k[cID], y, cmu, kappa)
        yplus = y_plus(k[cID], nu[cID], y, cmu) 

        if yplus > yPlusLam 
            ωc = ωlog
        else
            ωc = ωvis
        end
        Atomix.@atomic sums[cID] += ωc
        Atomix.@atomic counts[cID] += one(eltype(counts))
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

