export discretise!
export create_model

# Model definitions
@build_model "Diffusion" 1 1 # generates struct and constructor (named tutple)
export Diffusion
@discretise Diffusion 1 1 # custom discretisation function

@build_model "ConvectionDiffusion" 2 1
export ConvectionDiffusion
@discretise ConvectionDiffusion 2 1

# Model constructors

function create_model(::Type{ConvectionDiffusion}, U, J, phi, S)
    model = ConvectionDiffusion(
        Divergence{Linear}(U, phi),
        Laplacian{Linear}(J, phi),
        S
        )
    model.terms.term2.sign[1] = -1
    return model
end

function create_model(::Type{Diffusion}, J, phi, S)
    model = Diffusion(
        Laplacian{Linear}(J, phi),
        S
        )
    return model
end
