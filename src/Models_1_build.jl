export discretise!

# Model definitions
@build_model "SteadyDiffusion" 1 1 # generates struct and constructor (named tutple)
export SteadyDiffusion
@discretise SteadyDiffusion 1 1 # custom discretisation function

@build_model "SteadyConvectionDiffusion" 2 1
export SteadyConvectionDiffusion
@discretise SteadyConvectionDiffusion 2 1
