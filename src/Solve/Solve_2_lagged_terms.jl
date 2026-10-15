export LaggedTerms, extrapolate_lagged!

"""
    LaggedTerms(eqn, fields; evaluate=nothing)
    extrapolate_lagged!(lagged::LaggedTerms, config)

Explicitly treated (lagged) fields of a transient solver, such as the convecting face flux, the
effective viscosity and the velocity gradient, which the solver recomputes later in each step.
A `lag => field` pair is a lagged copy of a transported field. Called once at the start of each
step, `extrapolate_lagged!` sets them to the time level the time scheme of `eqn` integrates at:
Euler keeps their values from the end of the previous step; Crank-Nicolson extrapolates them to
tⁿ⁺¹ from the two previous steps, with weights from the time-step history, after `evaluate()`
has set them from the initial state on the first step.
"""
struct LaggedTerms{A,C,H,E,F}
    arrays::A       # value arrays of the lagged fields
    copies::C       # (copy, field) value arrays of the lagged copies
    previous::H     # values at the end of the step before (`nothing`: no extrapolation)
    current::H      # scratch
    evaluate::E
    dt::Vector{F}   # [Δt of the previous step, steps taken]
end

_lagged_arrays(f::Union{ScalarField,FaceScalarField}) = (f.values,)
_lagged_arrays(f::Union{VectorField,FaceVectorField}) = (f.x.values, f.y.values, f.z.values)
_lagged_arrays(f::TensorField) = (f.xx.values, f.xy.values, f.xz.values, f.yx.values,
    f.yy.values, f.yz.values, f.zx.values, f.zy.values, f.zz.values)
_lagged_arrays(f::ConstantScalar) = ()
_lagged_arrays(p::Pair) = _lagged_arrays(p.first)
_copy_arrays(f) = ()
_copy_arrays(p::Pair) = tuple(zip(_lagged_arrays(p.first), _lagged_arrays(p.second))...)

_extrapolates(eqn) = eqn.model.terms[1].type isa Time{CrankNicolson}

function LaggedTerms(eqn, fields; evaluate=nothing)
    arrays = foldl((a, f) -> (a..., _lagged_arrays(f)...), fields; init=())
    copies = foldl((a, f) -> (a..., _copy_arrays(f)...), fields; init=())
    previous, current = _extrapolates(eqn) ?
        (map(similar, arrays), map(similar, arrays)) : (nothing, nothing)
    LaggedTerms(arrays, copies, previous, current, evaluate, [NaN, 0.0])
end

_update_copies!(lagged) = foreach(((c, f),) -> copyto!(c, f), lagged.copies)

extrapolate_lagged!(lagged::LaggedTerms{A,C,Nothing}, config) where {A,C} = _update_copies!(lagged)

# xⁿ⁺¹ ≈ xⁿ + (Δtⁿ/Δtⁿ⁻¹)(xⁿ - xⁿ⁻¹); the first step keeps xⁿ
function extrapolate_lagged!(lagged::LaggedTerms, config)
    dt = Float64(Array(config.runtime.dt)[1])
    (; arrays, previous, current) = lagged
    lagged.dt[2] == 0 && !isnothing(lagged.evaluate) && lagged.evaluate()
    _update_copies!(lagged)
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
