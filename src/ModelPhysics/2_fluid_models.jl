export Incompressible
export _nu

@kwdef struct Incompressible{T}
    nu::T
end
Adapt.@adapt_structure Incompressible

_nu(fluid::Incompressible) = fluid.nu
# rho(fluid::Incompressible) = fluid.nu