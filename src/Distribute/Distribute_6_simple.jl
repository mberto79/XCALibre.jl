export psimple!, ppiso!

"""
    psimple!(model, config; petsc_options="", pref=nothing, ncorrectors=0, inner_loops=0)

Distributed steady incompressible SIMPLE solver: `simple!` on a `DistributedMesh` with
PETSc distributed solves. Laminar turbulence only in v1. Output deferred to Phase 8.
"""
psimple!(model, config; petsc_options="", pref=nothing, ncorrectors=0, inner_loops=0, kwargs...) =
    psetup_incompressible_solvers(PSIMPLE, model, config; petsc_options, pref, ncorrectors, inner_loops)

"""
    ppiso!(model, config; petsc_options="", pref=nothing, ncorrectors=0, inner_loops=2)

Distributed transient incompressible PISO solver (`piso!` counterpart of [`psimple!`](@ref)).
"""
ppiso!(model, config; petsc_options="", pref=nothing, ncorrectors=0, inner_loops=2, kwargs...) =
    psetup_incompressible_solvers(PPISO, model, config; petsc_options, pref, ncorrectors, inner_loops)

prun!(model::Physics{T,F,SO,M,Tu,E,D,BI}, config; petsc_options="", kwargs...
    ) where {T<:Steady,F<:Incompressible,SO,M,Tu,E,D<:DistributedMesh,BI} =
    psimple!(model, config; petsc_options, kwargs...)

prun!(model::Physics{T,F,SO,M,Tu,E,D,BI}, config; petsc_options="", kwargs...
    ) where {T<:Transient,F<:Incompressible,SO,M,Tu,E,D<:DistributedMesh,BI} =
    ppiso!(model, config; petsc_options, kwargs...)

# NEW SECTION: setup (mirrors setup_incompressible_solvers; PETSc PC owns preconditioning)

function psetup_incompressible_solvers(
    solver_variant, model, config; petsc_options="", pref=nothing, ncorrectors=0, inner_loops=0)
    (; solvers, schemes, hardware, boundaries) = config
    (; U, p) = model.momentum
    dmesh = model.domain
    dmesh isa DistributedMesh || error("distributed solvers require model.domain::DistributedMesh — build it with distribute(mesh)")
    model.turbulence isa Laminar || error("distributed v1 supports RANS{Laminar} only")
    _check_no_periodic(boundaries.U); _check_no_periodic(boundaries.p)

    ∇p = Grad{schemes.p.gradient}(p)
    mdotf = FaceScalarField(dmesh)
    rDf = FaceScalarField(dmesh)
    initialise!(rDf, 1.0)
    nueff = FaceScalarField(dmesh)
    divHv = ScalarField(dmesh)

    U_eqn = (
        Time{schemes.U.time}(U)
        + Divergence{schemes.U.divergence}(mdotf, U)
        - Laplacian{schemes.U.laplacian}(nueff, U)
        ==
        - Source(∇p.result)
    ) → VectorEquation(U, boundaries.U)

    p_eqn = (
        - Laplacian{schemes.p.laplacian}(rDf, p) == - Source(divHv)
    ) → ScalarEquation(p, boundaries.p)

    turbulenceModel, config = initialise(model.turbulence, model, mdotf, p_eqn, config)

    (; backend) = hardware
    U_deqn = DistributedEqn(U_eqn,
        PETScSolver(U_eqn, dmesh, solvers.U; petsc_options),
        dmesh.partition, HaloExchange(dmesh, 1, backend))
    p_deqn = DistributedEqn(p_eqn,
        PETScSolver(p_eqn, dmesh, solvers.p; petsc_options),
        dmesh.partition, HaloExchange(dmesh, 1, backend))

    solver_variant(model, turbulenceModel, ∇p, U_deqn, p_deqn, config;
        pref, ncorrectors, inner_loops)
end

_check_no_periodic(BCs) = begin
    any(BC isa PeriodicParent || BC isa Periodic for BC ∈ BCs) &&
        error("distributed v1 does not support periodic boundaries")
    nothing
end

# NEW SECTION: PSIMPLE loop (serial SIMPLE + halo syncs; see phase5.md sync map)

function PSIMPLE(model, turbulenceModel, ∇p, U_deqn, p_deqn, config;
    pref=nothing, ncorrectors=0, inner_loops=0)
    (; U, p, Uf, pf) = model.momentum
    (; nu) = model.fluid
    dmesh = model.domain
    (; solvers, schemes, boundaries) = config
    (; iterations) = config.runtime
    (; backend, workgroup) = config.hardware
    rank = dmesh.partition.rank

    U_eqn, p_eqn = U_deqn.eqn, p_deqn.eqn
    mdotf = get_flux(U_eqn, 2)
    nueff = get_flux(U_eqn, 3)
    rDf = get_flux(p_eqn, 1)
    divHv = get_source(p_eqn, 1)

    gradU = Grad{schemes.U.gradient}(U)
    gradUT = T(gradU)
    S = StrainRate(gradU, gradUT, U, Uf)

    Hv = VectorField(dmesh)
    rD = ScalarField(dmesh)

    TF = _get_float(dmesh)
    prev = KernelAbstractions.zeros(backend, TF, length(dmesh.cells))
    R_ux = zeros(TF, iterations)
    R_uy = zeros(TF, iterations)
    R_uz = zeros(TF, iterations)
    R_p = zeros(TF, iterations)

    H1 = p_deqn.halo
    H3 = HaloExchange(dmesh, 3, backend)

    time = zero(TF)
    halo_exchange!(U, H3, backend, workgroup)
    halo_exchange!(p, H1, backend, workgroup)
    interpolate!(Uf, U, config)
    correct_boundaries!(Uf, U, boundaries.U, time, config)
    flux!(mdotf, Uf, config)
    grad!(∇p, pf, p, boundaries.p, time, config)
    limit_gradient!(schemes.p.limiter, ∇p, p, config)
    halo_exchange!(∇p.result, H3, backend, workgroup)
    update_nueff!(nueff, nu, model.turbulence, config)

    xdir, ydir, zdir = XDir(), YDir(), ZDir()
    is3d = dmesh.mesh isa Mesh3

    for iteration ∈ 1:iterations
        time = iteration

        rx, ry, rz = solve_equation!(U_deqn, U, boundaries.U, solvers.U, xdir, ydir, zdir, config)

        inverse_diagonal!(rD, U_eqn, config)
        halo_exchange!(rD, H1, backend, workgroup)
        interpolate!(rDf, rD, config)
        remove_pressure_source!(U_eqn, ∇p, config)
        H!(Hv, U, U_eqn, config)
        halo_exchange!(Hv, H3, backend, workgroup)

        interpolate!(Uf, Hv, config)
        correct_boundaries!(Uf, Hv, boundaries.U, time, config)
        flux!(mdotf, Uf, config)
        div!(divHv, mdotf, config)

        @. prev = p.values
        rp = solve_equation!(p_deqn, p, boundaries.p, solvers.p, config; ref=pref)
        explicit_relaxation!(p, prev, solvers.p.relax, config)
        halo_exchange!(p, H1, backend, workgroup)

        grad!(∇p, pf, p, boundaries.p, time, config)
        limit_gradient!(schemes.p.limiter, ∇p, p, config)
        halo_exchange!(∇p.result, H3, backend, workgroup)

        for i ∈ 1:ncorrectors
            discretise!(p_eqn, p, config)
            apply_boundary_conditions!(p_eqn, boundaries.p, nothing, time, config)
            nonorthogonal_face_correction(p_eqn, ∇p, rDf, config)
            rp = solve_system!(p_deqn, solvers.p, p, nothing, config)
            explicit_relaxation!(p, prev, solvers.p.relax, config)
            halo_exchange!(p, H1, backend, workgroup)
            grad!(∇p, pf, p, boundaries.p, time, config)
            limit_gradient!(schemes.p.limiter, ∇p, p, config)
            halo_exchange!(∇p.result, H3, backend, workgroup)
        end

        pcorrect_mass_flux!(mdotf, p_eqn, config; time=time)
        correct_velocity!(U, Hv, ∇p, rD, config)

        turbulence!(turbulenceModel, model, S, prev, time, config)
        update_nueff!(nueff, nu, model.turbulence, config)

        R_ux[iteration] = rx
        R_uy[iteration] = ry
        R_uz[iteration] = rz
        R_p[iteration] = rp

        Uz_convergence = is3d ? rz <= solvers.U.convergence : true
        if (rx <= solvers.U.convergence && ry <= solvers.U.convergence &&
            Uz_convergence && rp <= solvers.p.convergence &&
            turbulenceModel.state.converged)
            rank == 0 && @info "Simulation converged in $iteration iterations!"
            break
        end
    end
    return (Ux=R_ux, Uy=R_uy, Uz=R_uz, p=R_p)
end

# NEW SECTION: PPISO loop (serial PISO + halo syncs)

function PPISO(model, turbulenceModel, ∇p, U_deqn, p_deqn, config;
    pref=nothing, ncorrectors=0, inner_loops=2)
    (; U, p, Uf, pf) = model.momentum
    (; nu) = model.fluid
    dmesh = model.domain
    (; solvers, schemes, boundaries) = config
    (; iterations) = config.runtime
    (; backend, workgroup) = config.hardware

    U_eqn, p_eqn = U_deqn.eqn, p_deqn.eqn
    mdotf = get_flux(U_eqn, 2)
    nueff = get_flux(U_eqn, 3)
    rDf = get_flux(p_eqn, 1)
    divHv = get_source(p_eqn, 1)

    gradU = Grad{schemes.U.gradient}(U)
    gradUT = T(gradU)
    S = StrainRate(gradU, gradUT, U, Uf)

    Hv = VectorField(dmesh)
    rD = ScalarField(dmesh)

    TF = _get_float(dmesh)
    n_cells = length(dmesh.cells)
    prev = KernelAbstractions.zeros(backend, TF, n_cells)
    cellsCourant = KernelAbstractions.zeros(backend, TF, n_cells)
    R_ux = ones(TF, iterations)
    R_uy = ones(TF, iterations)
    R_uz = ones(TF, iterations)
    R_p = ones(TF, iterations)
    dt_cpu = zeros(TF, 1)

    H1 = p_deqn.halo
    H3 = HaloExchange(dmesh, 3, backend)

    time = zero(TF)
    halo_exchange!(U, H3, backend, workgroup)
    halo_exchange!(p, H1, backend, workgroup)
    interpolate!(Uf, U, config)
    correct_boundaries!(Uf, U, boundaries.U, time, config)
    flux!(mdotf, Uf, config)
    grad!(∇p, pf, p, boundaries.p, time, config)
    limit_gradient!(schemes.p.limiter, ∇p, p, config)
    halo_exchange!(∇p.result, H3, backend, workgroup)
    update_nueff!(nueff, nu, model.turbulence, config)

    xdir, ydir, zdir = XDir(), YDir(), ZDir()

    for iteration ∈ 1:iterations
        copyto!(dt_cpu, config.runtime.dt)
        time += dt_cpu[1]

        rx, ry, rz = solve_equation!(
            U_deqn, U, boundaries.U, solvers.U, xdir, ydir, zdir, config; time=time)

        inverse_diagonal!(rD, U_eqn, config)
        halo_exchange!(rD, H1, backend, workgroup)
        interpolate!(rDf, rD, config)
        remove_pressure_source!(U_eqn, ∇p, config)

        rp = zero(TF)
        for i ∈ 1:inner_loops
            H!(Hv, U, U_eqn, config)
            halo_exchange!(Hv, H3, backend, workgroup)

            interpolate!(Uf, Hv, config)
            correct_boundaries!(Uf, Hv, boundaries.U, time, config)
            flux!(mdotf, Uf, config)
            div!(divHv, mdotf, config)

            @. prev = p.values
            rp = solve_equation!(p_deqn, p, boundaries.p, solvers.p, config; ref=pref, time=time)
            explicit_relaxation!(p, prev, i == inner_loops ? 1.0 : solvers.p.relax, config)
            halo_exchange!(p, H1, backend, workgroup)

            grad!(∇p, pf, p, boundaries.p, time, config)
            limit_gradient!(schemes.p.limiter, ∇p, p, config)
            halo_exchange!(∇p.result, H3, backend, workgroup)

            for j ∈ 1:ncorrectors
                discretise!(p_eqn, p, config)
                apply_boundary_conditions!(p_eqn, boundaries.p, nothing, time, config)
                setReference!(p_deqn, pref, 1, config)
                nonorthogonal_face_correction(p_eqn, ∇p, rDf, config)
                rp = solve_system!(p_deqn, solvers.p, p, nothing, config)
                explicit_relaxation!(p, prev, j == ncorrectors ? 1.0 : solvers.p.relax, config)
                halo_exchange!(p, H1, backend, workgroup)
                grad!(∇p, pf, p, boundaries.p, time, config)
                limit_gradient!(schemes.p.limiter, ∇p, p, config)
                halo_exchange!(∇p.result, H3, backend, workgroup)
            end

            pcorrect_mass_flux!(mdotf, p_eqn, config)
            correct_velocity!(U, Hv, ∇p, rD, config)
        end

        turbulence!(turbulenceModel, model, S, prev, time, config)
        update_nueff!(nueff, nu, model.turbulence, config)

        courant = pmax_courant_number!(cellsCourant, model, config, H1.comm)
        update_dt!(config.runtime, courant)

        R_ux[iteration] = rx
        R_uy[iteration] = ry
        R_uz[iteration] = rz
        R_p[iteration] = rp
    end
    return (Ux=R_ux, Uy=R_uy, Uz=R_uz, p=R_p)
end

# NEW SECTION: distributed helpers

# serial kernel reads A[owner1, owner2]; on processor faces owner1 may be the ghost, whose
# CSR row is garbage — read the smaller local id's row (owned block precedes ghosts);
# the interior Laplacian coefficient is symmetric, so the value is identical
function pcorrect_mass_flux!(mdotf, p_eqn, config; time=nothing)
    (; faces, boundary_cellsID) = mdotf.mesh
    (; backend, workgroup) = config.hardware
    p = p_eqn.model.terms[1].phi
    A = _A(p_eqn)
    n_bfaces = length(boundary_cellsID)
    ndrange = length(faces) - n_bfaces
    kernel! = _pcorrect_mass_flux!(_setup(backend, workgroup, ndrange)...)
    kernel!(mdotf, p, _nzval(A), _colval(A), _rowptr(A), faces, n_bfaces)
    KernelAbstractions.synchronize(backend)
    correct_boundary_mass_flux!(mdotf, p_eqn, config.boundaries.p, time, config)
end

@kernel function _pcorrect_mass_flux!(mdotf, p, nzval, colval, rowptr, faces, n_bfaces)
    i = @index(Global)
    fID = i + n_bfaces
    @inbounds begin
        (; ownerCells) = faces[fID]
        cID1 = ownerCells[1]
        cID2 = ownerCells[2]
        r = min(cID1, cID2)
        c = max(cID1, cID2)
        aN = nzval[spindex(rowptr, colval, r, c)]
        mdotf[fID] += aN*(p[cID2] - p[cID1])
    end
end

# _max_courant_number! dispatches on the wrapped Mesh2/Mesh3; global max keeps dt identical
function pmax_courant_number!(cellsCourant, model, config, comm)
    (; U) = model.momentum
    (; backend, workgroup) = config.hardware
    ndrange = length(cellsCourant)
    kernel! = _max_courant_number!(_setup(backend, workgroup, ndrange)...)
    kernel!(cellsCourant, U, config.runtime, model.domain.mesh)
    MPI.Allreduce(maximum(cellsCourant), max, comm)
end
