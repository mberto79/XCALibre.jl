export filmModel!

function filmModel!(
    model, config;
    output=VTK()#, pref=nothing, ncorrectors=0, inner_loops=0
)
    print("Using film model\n")
    residuals = setup_FilmModel_Solver(
        FilmModel, model, config,
        output=output
    )
    
    return residuals
end

function setup_FilmModel_Solver(solver_variant, model, config;
    output=VTK())

    (; solvers, schemes, runtime, hardware, boundaries) = config

    @info "Extracting configuration and input fields..."

    (; U, h, Uf, hf) = model.momentum
    mesh = model.domain
    (; rho) = model.fluid
    rho_l = rho

    @info "Pre-allocating fields..."
    ∇p = Grad{schemes.h.gradient}(h)
    mdotf = FaceScalarField(mesh)
    Sm = ScalarField(mesh)
    Si_mom = ScalarField(mesh)
    nueff = FaceScalarField(mesh)

    @info "Defining models.."

    U_eqn = (
        Time{schemes.U.time}(U)
        + Divergence{schemes.U.divergence}(mdotf,U)
        - Laplacian{schemes.U.laplacian}(nueff, U)
        ==
        Source(∇p.result)
    ) → VectorEquation(U, boundaries.U)

    h_eqn = (
        Time{schemes.h.time}(rho_l,h)
        + Divergence{schemes.h.divergence}(mdotf, h)
        ==
        Source(Sm)
    ) → ScalarEquation(h, boundaries.h)

    @info "Initialising preconditioners"

    @reset U_eqn.preconditioner = set_preconditioner(solvers.U.preconditioner, U_eqn)
    @reset h_eqn.preconditioner = set_preconditioner(solvers.h.preconditioner, h_eqn)

    @info "Pre-allocating solvers"

    @reset U_eqn.solver = _workspace(solvers.U.solver, _b(U_eqn, XDir()))
    @reset h_eqn.solver = _workspace(solvers.h.solver, _b(h_eqn))
end
