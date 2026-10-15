export time_level!

# Old time levels of a Crank-Nicolson Time term, ddtⁿ⁺¹ = (1 + c)(ρxⁿ⁺¹ - ρxⁿ)/Δt - c ddtⁿ.
# The operator advances them itself when the time passed to `discretise!` changes, so the
# solvers need no knowledge of the scheme.
struct CNState{F,A,S,V,C}
    ψ::F
    old::A      # ρxⁿ, one array per component
    ddt0::A     # ddtⁿ
    source::S   # S = (1 + c)ρxⁿ/Δt + c ddtⁿ, the source per unit volume (a field, read as `prev`)
    factor::V   # [1 + c] on the device
    clock::C    # host: time and Δt of the current step and its c (NaN before the first step)
end
Adapt.@adapt_structure CNState

# host-side record of the current step; kernels receive the operator without it (`_kernel_type`)
struct HostClock
    v::Vector{Float64}
end
Adapt.adapt_structure(to, c::HostClock) = c

_level_like(phi::ScalarField, mesh) = ScalarField(mesh)
_level_like(phi::VectorField, mesh) = VectorField(mesh)
_components(f::ScalarField) = (f.values,)
_components(f::VectorField) = (f.x.values, f.y.values, f.z.values)
_rho_values(r::ConstantScalar) = r.values
_rho_values(r) = r.values

function _cn_operator(ψ, phi)
    mesh = phi.mesh
    source = _level_like(phi, mesh)
    arrays() = map(similar, _components(source))
    F = _get_float(mesh)
    backend = _get_backend(mesh)
    factor = KernelAbstractions.ones(backend, F, 1)
    state = CNState(F(ψ), arrays(), arrays(), source, factor, HostClock([NaN, NaN, NaN]))
    Time{CrankNicolson,typeof(state)}(state)
end
ModelFramework.time_operator(T::Type{CrankNicolson}, phi) = _cn_operator(crank_nicolson_coefficient(T), phi)
ModelFramework.time_operator(T::CrankNicolson, phi) = _cn_operator(crank_nicolson_coefficient(T), phi)

# kernels read only 1 + c from a Crank-Nicolson Time term
_kernel_type(t) = t
_kernel_type(t::Time{CrankNicolson,<:CNState}) = Time{CrankNicolson,typeof(t.state.factor)}(t.state.factor)

"""
    time_level!(term, prev, rho_prev, time, config)

Returns the field the Time term reads as its old level when the equation is discretised at
`time`. A Crank-Nicolson term advances its stored levels on the first call of a new time and
returns its source field; other schemes return `prev`.
"""
time_level!(term, prev, rho_prev, time, config) = prev

function time_level!(
    term::Operator{F,P,I,<:Time{CrankNicolson,<:CNState}}, prev, rho_prev, time, config
    ) where {F,P,I}
    isnothing(time) && throw(ArgumentError("a CrankNicolson Time term needs the time of the step"))
    s = term.type.state
    clock = s.clock.v
    time == clock[1] && return s.source # same step: same old level
    dt = Float64(Array(config.runtime.dt)[1]) # once per step
    r = _rho_values(rho_prev)
    if isnan(clock[1]) # first step of the run: backward Euler
        foreach((o, x) -> (@. o = r*x), s.old, _components(prev))
        foreach(d -> fill!(d, zero(eltype(d))), s.ddt0)
        c = 0.0
    else # ddtⁿ of the completed step, then ρxⁿ
        c0, dt0 = clock[3], clock[2]
        foreach(s.ddt0, s.old, _components(prev)) do d, o, x
            @. d = (1 + c0)*(r*x - o)/dt0 - c0*d
            @. o = r*x
        end
        c = Float64(s.ψ)
    end
    clock .= (time, dt, c)
    fill!(s.factor, 1 + c)
    foreach(_components(s.source), s.old, s.ddt0) do b, o, d
        @. b = (1 + c)*o/dt + c*d
    end
    s.source
end
