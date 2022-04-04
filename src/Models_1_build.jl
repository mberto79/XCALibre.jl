export discretise!
export discretise2!
export discretise3!
export discretise4!

# Model definitions
@build_model "SteadyDiffusion" 1 1 # generates struct and constructor (named tutple)
export SteadyDiffusion
@discretise SteadyDiffusion 1 1 # custom discretisation function
@discretise2 SteadyDiffusion 1 1 # custom discretisation function

@build_model "SteadyConvectionDiffusion" 2 1
export SteadyConvectionDiffusion
@discretise SteadyConvectionDiffusion 2 1
@discretise2 SteadyConvectionDiffusion 2 1
@discretise3 SteadyConvectionDiffusion 2 1
@discretise4 SteadyConvectionDiffusion 2 1
