# Gradient of a transported scalar for second-order convection schemes (see `LinearUpwind`).
# It is only allocated when the scheme asks for it; every other scheme gets `nothing` and no work.

struct ConvectionGradient{G,F}
    grad::G # Grad of the transported field
    phif::F # face values (scratch for the gradient calculation)
end
Adapt.@adapt_structure ConvectionGradient

convection_gradient(scheme, phi, gradient_scheme) =
    is_second_order(scheme) ? ConvectionGradient(
        Grad{gradient_scheme}(phi), FaceScalarField(phi.mesh)) : nothing

# gradient argument of Divergence{scheme}(flux, phi, grad)
convection_grad_field(::Nothing) = nothing
convection_grad_field(cg::ConvectionGradient) = cg.grad.result

update_convection_gradient!(::Nothing, phi, BCs, limiter, time, config) = nothing
function update_convection_gradient!(cg::ConvectionGradient, phi, BCs, limiter, time, config)
    grad!(cg.grad, cg.phif, phi, BCs, time, config)
    limit_gradient!(limiter, cg.grad, phi, config)
    nothing
end
