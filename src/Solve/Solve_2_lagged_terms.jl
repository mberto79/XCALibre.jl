export LaggedTerms, extrapolate_lagged!

"""
    LaggedTerms(eqn, fields)
    extrapolate_lagged!(lagged::LaggedTerms, config)

Explicitly treated (lagged) fields of a transient solver, such as the convecting face flux, the
effective viscosity and the velocity gradient, which the solver recomputes later in each step.
Called once at the start of each step, `extrapolate_lagged!` sets them to the time level the
time scheme of `eqn` integrates at: Euler keeps their values from the end of the previous
step; Crank-Nicolson extrapolates them to tⁿ⁺¹ from the two previous steps, with weights from
the time-step history.
"""
struct LaggedTerms{A,H,F}
    arrays::A       # value arrays of the lagged fields
    previous::H     # their values at the end of the step before (`nothing`: no extrapolation)
    current::H      # scratch
    dt::Vector{F}   # [Δt of the previous step, steps taken]
end

_lagged_arrays(f::Union{ScalarField,FaceScalarField}) = (f.values,)
_lagged_arrays(f::Union{VectorField,FaceVectorField}) = (f.x.values, f.y.values, f.z.values)
_lagged_arrays(f::TensorField) = (f.xx.values, f.xy.values, f.xz.values, f.yx.values,
    f.yy.values, f.yz.values, f.zx.values, f.zy.values, f.zz.values)
_lagged_arrays(f::ConstantScalar) = ()

_extrapolates(eqn) = eqn.model.terms[1].type isa Time{CrankNicolson}

function LaggedTerms(eqn, fields)
    arrays = foldl((a, f) -> (a..., _lagged_arrays(f)...), fields; init=())
    previous, current = _extrapolates(eqn) ?
        (map(similar, arrays), map(similar, arrays)) : (nothing, nothing)
    LaggedTerms(arrays, previous, current, [NaN, 0.0])
end

extrapolate_lagged!(lagged::LaggedTerms{A,Nothing}, config) where A = nothing

# xⁿ⁺¹ ≈ xⁿ + (Δtⁿ/Δtⁿ⁻¹)(xⁿ - xⁿ⁻¹); the first step keeps xⁿ
function extrapolate_lagged!(lagged::LaggedTerms, config)
    dt = Float64(Array(config.runtime.dt)[1])
    (; arrays, previous, current) = lagged
    foreach(copyto!, current, arrays)
    if lagged.dt[2] > 0
        r = dt/lagged.dt[1]
        foreach(arrays, previous) do x, x0
            rx = eltype(x)(r)
            @. x = x + rx*(x - x0)
        end
    end
    foreach(copyto!, previous, current)
    lagged.dt .= (dt, lagged.dt[2] + 1)
    nothing
end
