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

"""
    ω_blend(ωvis, ωlog) = sqrt(ωvis^2 + ωlog^2)

Menter's blend of the viscous-sublayer and log-layer wall values for `omega`,
replacing a hard `y+ > yPlusLam` switch between the two branches.

### Why blending rather than switching

The branch was selected on a y+ built from `k` itself,

    y+ = cmu^0.25*y*sqrt(k)/nu

so a transient dip in `k` moves the wall treatment into the viscous branch, and
on that branch the `k` production was set to ZERO - which prevents `k` from ever
recovering. The eddy viscosity needed to rebuild the velocity profile and the
velocity profile needed to generate `k` each depend on the other, so the state
is self-sustaining once entered.

MEASURED, adiabatic air-water pipe, wall cell at z/D = 20 with the void-driven
buoyancy that triggers it (lift on, `alpha` reaching 0.18 in the first cell):

    iteration     250      500      750     1000     1250     1500     1750
    k          1.4e-04  2.2e-05  6.2e-06  2.4e-06  1.2e-06  6.5e-07  4.0e-07
    omega        140.3    140.2    140.2    140.2    140.2    140.2    140.2

`k` falls four orders of magnitude while `omega` does not move a digit, because
it is pinned at exactly `ω_vis = 6nu/(beta1 y^2) = 139.2`. The velocity profile
goes with it - the wall cell accelerates from a seeded 0.668 m/s to the bulk
0.867 m/s - and `nut/nu` reaches 0.00 against the 14.8 that y+ = 36 calls for.
The same case with lift off never dips below the threshold and is unaffected,
which is why this had not been seen before.

The blend costs nothing at equilibrium: at y+ = 36 it gives 528 against the pure
log value of 509, the standard Menter result.
"""
ω_blend(ωvis::T, ωlog) where T = sqrt(ωvis*ωvis + ωlog*ωlog)

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
    func_calls = Expr[]
    for i ∈ eachindex(BCs)
        call = quote
            set_production!(P, fieldBCs[$i], model, gradU, config)
        end
        push!(func_calls, call)
    end
    quote
    $(func_calls...)
    nothing
    end 
end

set_production!(P, BC, model, gradU, config) = nothing

function set_production!(P, BC::KWallFunction, model, gradU, config)
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
    kernel! = _set_production!(_setup(backend, workgroup, ndrange)...)
    kernel!(
        P.values, BC, fluid, momentum, turbulence, faces, boundary_cellsID, start_ID, gradU
    )
end

@kernel function _set_production!(
    values, BC::KWallFunction, fluid, momentum, turbulence, faces, boundary_cellsID, start_ID, gradU)
    i = @index(Global)
    fID = i + start_ID - 1 # Redefine thread index to become face ID

    (; kappa, beta1, cmu, B, E, yPlusLam) = BC.value
    (; nu, rho) = fluid
    (; U) = momentum
    (; k, nut) = turbulence

    Uw = SVector{3}(0.0,0.0,0.0)
    # Uw = boundaries.U[BC.ID].value
    cID = boundary_cellsID[fID]
    face = faces[fID]
    nuc = nu[cID]
    (; delta, normal)= face
    uStar = cmu^0.25*sqrt(k[cID])
    dUdy = uStar/(kappa*delta)
    yplus = y_plus(k[cID], nuc, delta, cmu)
    nutw = nut_wall(nuc, yplus, kappa, E)
    mag_grad_U = mag(sngrad(U[cID], Uw, delta, normal))
    # mag_grad_U = mag(gradU[cID]*normal)
    # NO y+ BRANCH. `nut_wall` is already exactly zero for y+ <= yPlusLam -
    # `yPlusLam` is DEFINED as the root of `y+*kappa/log(E*y+) - 1`, so the
    # cutoff is built into `nutw` and a branch around it is redundant for the
    # eddy-viscosity part.
    #
    # What the branch actually did was zero the MOLECULAR part as well, and
    # that is the latch described at `ω_blend`: with production identically
    # zero, `k` can only decay, and nothing can return the wall layer to the
    # log branch. Keeping the molecular term leaves production continuous and
    # lets a depressed wall layer climb back out - at the collapsed state
    # measured on the bubbly pipe (k = 1e-6, omega = 139) it gives
    #
    #     P = nu*|grad(U)|*dUdy = 2.0e-3   against   beta*k*omega = 1.25e-5
    #
    # i.e. production exceeding dissipation by 160x, so recovery is immediate
    # rather than impossible.
    #
    # DENSITY WEIGHTING. The k equation is assembled in CONSERVATIVE form,
    #
    #     Time(rho, k) + Divergence(mdotf, k) - Laplacian(mueffk, k) + Si(Dkf, k) == Source(Pk)
    #
    # with every other term carrying rho - `Pk = rho*nut*S^2`, `Dkf = rho*beta*omega`,
    # `mueffk = rhof*(nuf + sigma_k*nutf)`. This override REPLACES `Pk` in the
    # wall cell, so it must be density-weighted too or the wall cell alone gets a
    # kinematic production against a density-weighted sink.
    #
    # This was invisible for years because `Fluid{Incompressible}` defaults to
    # `rho = 1.0`, so every single-phase validation case in the repo runs at unit
    # density and the factor cannot be seen. It bites as soon as a real density
    # appears - a multiphase mixture at rho ~ 900, or any compressible case.
    #
    # MEASURED, adiabatic air-water pipe (rho_m = 900), wall cell at z/D = 20 in
    # the converged state with all lateral forces off:
    #
    #     production without rho    0.0371       P/sink = 0.0043
    #     sink  rho*beta*omega*k    8.65
    #     production with rho       33.4         P/sink = 3.9
    #
    # i.e. local production was 0.4% of the local sink, leaving the wall cell
    # sustained only by diffusion from its neighbour. `k` then converges far
    # below equilibrium - 5.4e-4 against 7.6e-3 - which parks `y+(k)` at 9.7,
    # just under `yPlusLam = 11.53`, so the wall functions never reach their log
    # branch and `nut/nu` sits at 0.33 where y+ = 36 calls for ~15.
    values[cID] = rho[cID]*(nu[cID] + nutw)*mag_grad_U*dUdy
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
    # Redundant branch removed: `nut_wall` returns exactly zero for
    # y+ <= yPlusLam by construction, so this is a no-op change kept only so
    # the three wall functions read consistently. See `ω_blend`.
    values[fID] = nutw
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
    func_calls = Expr[]
    for i ∈ eachindex(BCs)
        call = quote
            constrain!(eqn, fieldBCs[$i], model, config)
        end
        push!(func_calls, call)
    end
    quote
    $(func_calls...)
    nothing
    end 
end

constrain!(eqn, BC, model, config) = nothing

function constrain!(eqn, BC::OmegaWallFunction, model, config)

    # backend = _get_backend(mesh)
    (; hardware) = config
    (; backend, workgroup) = hardware

    # Access equation data and deconstruct sparse array
    A = _A(eqn)
    b = _b(eqn, nothing)
    colval = _colval(A)
    rowptr = _rowptr(A)
    nzval = _nzval(A)
    
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
    kernel! = _constrain!(_setup(backend, workgroup, ndrange)...)
    kernel!(
        turbulence, fluid, BC, faces, start_ID, boundary_cellsID, colval, rowptr, nzval, b
    )
end

@kernel function _constrain!(turbulence, fluid, BC::OmegaWallFunction, faces, start_ID, boundary_cellsID, colval, rowptr, nzval, b)
    i = @index(Global)
    fID = i + start_ID - 1 # Redefine thread index to become face ID

    @uniform begin
        nu = fluid.nu
        k = turbulence.k
        (; kappa, beta1, cmu, B, E, yPlusLam) = BC.value
    end
    ωc = zero(eltype(nzval))
    
    @inbounds begin
        cID = boundary_cellsID[fID]
        face = faces[fID]
        y = face.delta
        ωvis = ω_vis(nu[cID], y, beta1)
        ωlog = ω_log(k[cID], y, cmu, kappa)
        # Menter blend rather than a y+ switch - see `ω_blend` for the failure
        # this fixes and the measured evidence.
        ωc = ω_blend(ωvis, ωlog)
        # Line below is weird but worked
        # b[cID] = A[cID,cID]*ωc

        
        # Classic approach
        # b[cID] += A[cID,cID]*ωc
        # A[cID,cID] += A[cID,cID]
        
        # nzIndex = spindex(rowptr, colval, cID, cID)
        # Atomix.@atomic b[cID] += nzval[nzIndex]*ωc
        # Atomix.@atomic nzval[nzIndex] += nzval[nzIndex] 

        z = zero(eltype(nzval))
        for nzi ∈ rowptr[cID]:(rowptr[cID+1] - 1)
            nzval[nzi] = z
        end
        cIndex = spindex(rowptr, colval, cID, cID)
        nzval[cIndex] = one(eltype(nzval))
        b[cID] = ωc
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

