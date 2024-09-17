export Configuration
export set_hardware

"""
    @kwdef struct Configuration{SC,SL,RT,HW}
        schemes::SC
        solvers::SL
        runtime::RT
        hardware::HW
    end

The `Configuration` type is pass to all flow solvers and provides all the relevant configuration to run a simulation. 

# Inputs 

* `schemes::NamedTuple` this keyword argument is used to pass distretisation scheme information to flow solvers. See [Numerical setup](@ref) for details.
* `solvers::NamedTuple` this keyword argument is used to pass the configurations for the linear solvers for each field information to flow solvers. See [Runtime and solvers](@ref) for details.
* `runtime::NamedTuple` this keyword argument is used to pass runtime information to the flow solvers. See [Runtime and solvers](@ref) for details.
* `hardware::NamedTuple` this keyword argument is used to pass the hardware configuration and backend settings to the flow solvers. See [Pre-processing](@ref) for details.

# Example

```julia
config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)
```
"""
@kwdef struct Configuration{SC,SL,RT,HW}
    schemes::SC
    solvers::SL
    runtime::RT
    hardware::HW
end
Adapt.@adapt_structure Configuration

set_hardware(;backend, workgroup) = begin
    (backend=backend, workgroup=workgroup)
end
