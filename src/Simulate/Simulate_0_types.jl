export Configuration
export Hardware

"""
    @kwdef struct Configuration{SC,SL,RT,HW,PP}
        schemes::SC
        solvers::SL
        runtime::RT
        hardware::HW
        postprocess::PP = nothing
    end

The `Configuration` type is passed to all flow solvers and provides all the relevant information to run a simulation. 

# Inputs 

- `schemes::NamedTuple` this keyword argument is used to pass distretisation scheme information to flow solvers. See [Numerical setup](@ref) for details.
- `solvers::NamedTuple` this keyword argument is used to pass the configurations for the linear solvers for each field information to flow solvers. See [Runtime and solvers](@ref) for details.
- `runtime::NamedTuple` this keyword argument is used to pass runtime information to the flow solvers. See [Runtime and solvers](@ref) for details.
- `hardware::NamedTuple` this keyword argument is used to pass the hardware configuration and backend settings to the flow solvers. See [Pre-processing](@ref) for details.

# Optional keywords

- `postprocess` this keyword argument is used to pass any fields that need to be post-processed. See [Post-processing](@ref) for details.
# Example

```julia
config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)
```
"""
@kwdef struct Configuration{SC,SL,RT,HW,BC,PP}
    schemes::SC
    solvers::SL
    runtime::RT
    hardware::HW
    boundaries::BC
    postprocess::PP = nothing
end
Adapt.@adapt_structure Configuration


"""
    hardware = Hardware(backend, workgroup)

Struct used to configure the backend.

# Inputs

- `backend`: used to specify the backend e.g. `CPU()`, `CUDABackend()` or other backends supported by `KernelAbstraction.jl`
- `workgroup::Int` this is an integer specifying the number of workers that cooperate in a parallel run. For GPUs this could be set to the size of the device's warp e.g. `workgroup = 32`. On CPUs, the default value in `KernelAbstractions.jl` is currently `workgroup = 1024`.

# Output

This function returns a `Hardware` object with the fields `backend` and `workgroup` which are accessed by internally in `XCALibre.jl` to execute a given kernel in the target `backend`.
"""
@kwdef struct Hardware{B,W}
    backend::B
    workgroup::W
end
Adapt.@adapt_structure Hardware