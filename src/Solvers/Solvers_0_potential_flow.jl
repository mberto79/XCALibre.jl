export potential_flow!

_potential_boundary(bc::AbstractDirichlet, value) =
    Dirichlet(bc.ID, value, bc.IDs_range)
_potential_boundary(bc::Periodic, value) = bc
_potential_boundary(bc::PeriodicParent, value) = bc
_potential_boundary(bc, value) = Zerogradient(bc.ID, value, bc.IDs_range)

"""
    potential_flow!(model, config; ncorrectors=0, pref=nothing, time=0)

Project the current velocity field onto a divergence-free potential-flow field.
Velocity boundary conditions supply the initial face flux. Velocity-potential
boundary conditions are inferred from pressure: fixed pressure becomes fixed
zero potential, periodic patches remain periodic, and other patches use zero
normal gradient.

The corrected face-volume flux is returned with the linear-solver residual.
"""
function potential_flow!(model, config; ncorrectors=0, pref=nothing, time=0)
    ncorrectors >= 0 || throw(ArgumentError("ncorrectors must be non-negative"))

    mesh = model.domain
    (; U, Uf) = model.momentum
    (; schemes, solvers, boundaries) = config
    TF = _get_float(mesh)
    time_value = TF(time)

    potential_BCs = map(boundaries.p) do bc
        _potential_boundary(bc, zero(TF))
    end
    potential_boundaries = (; boundaries..., p=potential_BCs)
    potential_config = @set config.boundaries = potential_boundaries

    Phi = ScalarField(mesh)
    divphi = ScalarField(mesh)
    phif = FaceScalarField(mesh)
    unit_flux = ConstantScalar(one(TF))
    Phi_eqn = (
        -Laplacian{schemes.p.laplacian}(unit_flux, Phi) == -Source(divphi)
    ) → ScalarEquation(Phi, potential_BCs)

    @reset Phi_eqn.preconditioner = set_preconditioner(
        solvers.p.preconditioner, Phi_eqn)
    @reset Phi_eqn.solver = _workspace(solvers.p.solver, _b(Phi_eqn))

    interpolate!(Uf, U, potential_config)
    correct_boundaries!(Uf, U, boundaries.U, time_value, potential_config)
    flux!(phif, Uf, potential_config)
    div!(divphi, phif, potential_config)

    has_fixed_potential = any(bc -> bc isa Dirichlet, potential_BCs)
    reference = isnothing(pref) && !has_fixed_potential ? zero(TF) : pref
    previous = similar(Phi.values)
    @. previous = Phi.values
    nonorthogonal_flux = ncorrectors > 0 ? FaceScalarField(mesh) : nothing

    residual = solve_equation!(
        Phi_eqn, Phi, potential_BCs, solvers.p, potential_config;
        ref=reference, time=time_value,
    )

    gradPhi = Grad{schemes.p.gradient}(Phi)
    Phi_face = FaceScalarField(mesh)
    for _ in 1:ncorrectors
        grad!(gradPhi, Phi_face, Phi, potential_BCs, time_value, potential_config)
        limit_gradient!(schemes.p.limiter, gradPhi, Phi, potential_config)
        discretise!(Phi_eqn, Phi, potential_config)
        apply_boundary_conditions!(
            Phi_eqn, potential_BCs, nothing, time_value, potential_config)
        setReference!(Phi_eqn, reference, 1, potential_config)
        nonorthogonal_face_correction(
            Phi_eqn, gradPhi, unit_flux, potential_config;
            correction=nonorthogonal_flux)
        update_preconditioner!(Phi_eqn.preconditioner, mesh, potential_config)
        residual = solve_system!(
            Phi_eqn, solvers.p, Phi, nothing, potential_config)
    end

    correct_mass_flux!(
        phif, Phi_eqn, potential_config;
        previous=previous, time=time_value,
        nonorthogonal=nonorthogonal_flux,
    )
    reconstruct!(U, phif, potential_config)
    interpolate!(Uf, U, potential_config)
    correct_boundaries!(Uf, U, boundaries.U, time_value, potential_config)

    return (; residual, flux=phif, potential=Phi)
end
