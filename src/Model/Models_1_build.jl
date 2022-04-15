export discretise!

# Model definitions
@build_model "Diffusion" 1 1 # generates struct and constructor (named tutple)
export Diffusion
@discretise Diffusion 1 1 # custom discretisation function

@build_model "ConvectionDiffusion" 2 1
export ConvectionDiffusion
@discretise ConvectionDiffusion 2 1
