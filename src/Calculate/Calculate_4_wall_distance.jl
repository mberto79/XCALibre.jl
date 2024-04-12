export calc_wall_distance!
export residual!

function calc_wall_distance!(model, config)
    @info "Calculating wall distance..."

    (;mesh, phi, y) = model
    (; solvers, schemes, runtime) = config
    (; iterations) = runtime

    phif = FaceScalarField(mesh)
    initialise!(phif, 1.0)

    phi_eqn = (
        Laplacian{schemes.phi.laplacian}(phif, phi) == Source(ConstantScalar(-1.0))
    ) → Equation(mesh)

    @reset phi_eqn.preconditioner = set_preconditioner(
        solvers.phi.preconditioner, phi_eqn, phi.BCs, runtime)

    @reset phi_eqn.solver = solvers.phi.solver(_A(phi_eqn), _b(phi_eqn))

    phiGrad = Grad{schemes.phi.gradient}(phi)
    phif = FaceScalarField(mesh)
    grad!(phiGrad, phif, phi, phi.BCs)

    n_cells = length(mesh.cells)
    TF = _get_float(mesh)
    prev = zeros(TF, n_cells)
    R_phi = ones(TF, iterations)

    for iteration ∈ 1:iterations
        @. prev = phi.values
        discretise!(phi_eqn, prev, runtime)
        apply_boundary_conditions!(phi_eqn, phi.BCs)
        update_preconditioner!(phi_eqn.preconditioner)
        run!(phi_eqn, solvers.phi)
        explicit_relaxation!(phi, prev, solvers.phi.relax)
        residual!(R_phi, phi_eqn.equation, phi, iteration)
        grad!(phiGrad, phif, phi, phi.BCs)

        if R_phi[iteration] < solvers.phi.convergence
            @info "Wall distance converged!"
            break
        end
    end 
    wallDist = normalDistance(phi, phiGrad)
    y.values .= wallDist.values
    return model
end

function normalDistance(phi, phiGrad)
    wallDist = deepcopy(phi)
    @inbounds for i ∈ 1:length(phi.values)
        gradMag = norm(phiGrad.result[i])
        wallDist.values[i] = -gradMag + sqrt(gradMag^2 + 2.0*phi.values[i])
    end
    return wallDist
end

function residual!(Residual, equation, phi, iteration)
    (; A, b, R, Fx) = equation
    values = phi.values
    mul!(Fx, A, values)
    @inbounds @. R = abs(Fx - b)^2
    res = sqrt(mean(R))/norm(b)
    Residual[iteration] = res
    nothing
end

#=
using ProgressMeter
progress = Progress(iterations; dt=1.0, showspeed=true)
        ProgressMeter.next!(
            progress, showvalues = [
                (:phi, R_phi[iteration])]
            )
=#