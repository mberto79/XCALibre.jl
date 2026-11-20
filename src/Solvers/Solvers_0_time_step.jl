# NEW SECTION: time-step constraint contract

# Courant criterion: courant_rate!(rate, criterion, model, config) fills the cell rates λ (1/s),
# so that the cell Courant number is λ·dt. Solver-specific criteria live with their solver.
abstract type CourantCriterion end

# Time-step limit: max_time_step(limit, model, config) returns this rank's upper bound on dt
# (Inf for none). Limits apply in every time-stepping mode, fixed dt included.
abstract type TimeStepLimit end

# Courant number a criterion is held to under AdaptiveTimeStepping
target_courant(::CourantCriterion, adaptive) = adaptive.maxCo

# Mode (runtime.adaptive): update_dt!(ctrl, mode, model, config, iteration) sets, with
# set_time_step!, the dt of `iteration` from the Courant numbers of ctrl

# NEW SECTION: convective Courant number |U|/Δx, Δx = V^(1/d)

struct ConvectiveCo <: CourantCriterion end

function courant_rate!(rate, ::ConvectiveCo, model, config)
    (; backend, workgroup) = config.hardware
    kernel! = _sized(_convective_rate!, backend, workgroup, length(rate))
    kernel!(rate, model.momentum.U, _base_mesh(model.domain))
end

@kernel function _convective_rate!(rate, U, mesh)
    i = @index(Global)
    @uniform cells = mesh.cells
    @inbounds rate[i] = norm(U[i]) / _cell_length(cells[i].volume, mesh)
end

@inline _cell_length(volume, ::Mesh2) = sqrt(volume)
@inline _cell_length(volume, ::Mesh3) = volume^(one(volume)/typeof(volume)(3))

# NEW SECTION: time-step control

# The host copy `dt` is the global time step (the time increment under LocalTimeStepping), so
# solvers never read `runtime.dt` from the device; `rate` holds the cell time steps under LTS.
mutable struct TimeStepControl{C<:Tuple,L<:Tuple,A,V<:AbstractVector,TF<:AbstractFloat,N}
    const criteria::C
    const limits::L
    const adaptive::A
    const explicit::Bool
    const monitor::Bool
    const rate::V
    dt::TF
    max_rate::NTuple{N,TF}
    courant::NTuple{N,TF}
end

# explicit=true: dt is never held above the stability limit by minShrink, and LTS is allowed;
# monitor=false: no Courant numbers under fixed dt (nothing to adapt, nothing reported)
function TimeStepControl(criteria::Tuple, model, config; limits::Tuple=(), explicit::Bool=false, monitor::Bool=true)
    (; runtime, hardware) = config
    _check_mode(runtime.adaptive, criteria, limits, explicit)
    TF = _get_float(model.domain)
    rate = KernelAbstractions.zeros(hardware.backend, TF, length(model.domain.cells))
    dt = TF(Array(runtime.dt)[1]) # the only device-to-host copy of dt
    zs = map(_ -> zero(TF), criteria)
    ctrl = TimeStepControl(criteria, limits, runtime.adaptive, explicit, monitor, rate, dt, zs, zs)
    _limit_dt!(ctrl, model, config)
    return ctrl
end

_check_mode(adaptive, criteria, limits, explicit) = nothing
function _check_mode(::LocalTimeStepping, criteria, limits, explicit)
    explicit || throw(ArgumentError("LocalTimeStepping is only supported by the explicit density-based solver"))
    length(criteria) == 1 && isempty(limits) ||
        throw(ArgumentError("LocalTimeStepping takes one Courant criterion and no time-step limits"))
    return nothing
end

time_step(ctrl::TimeStepControl) = ctrl.dt
courant_numbers(ctrl::TimeStepControl) = ctrl.courant

# step of the explicit update: per cell under LocalTimeStepping, else the global dt
cell_time_step(ctrl::TimeStepControl) = _cell_time_step(ctrl, ctrl.adaptive)
_cell_time_step(ctrl, ::LocalTimeStepping) = ctrl.rate
_cell_time_step(ctrl, adaptive) = ctrl.dt

# Courant numbers at the current dt, then the dt of `iteration` from the mode and the limits
function update_time_step!(ctrl::TimeStepControl, model, config, iteration::Integer)
    if ctrl.monitor || !isnothing(ctrl.adaptive)
        ctrl.max_rate = map(c -> _max_rate!(ctrl.rate, c, model, config), ctrl.criteria)
        ctrl.courant = map(λ -> λ * ctrl.dt, ctrl.max_rate)
    end
    update_dt!(ctrl, ctrl.adaptive, model, config, iteration)
    _limit_dt!(ctrl, model, config)
    return ctrl
end

function _max_rate!(rate, criterion, model, config)
    courant_rate!(rate, criterion, model, config)
    mesh = model.domain
    return global_max(maximum(_owned(rate, mesh)), mesh)
end

# owned entries of a per-cell array (ghost entries come from partial face lists)
_owned(v, mesh) = view(v, 1:n_owned_cells(mesh))

update_dt!(ctrl, ::Nothing, model, config, iteration) = nothing

function update_dt!(ctrl, a::AdaptiveTimeStepping, model, config, iteration)
    TF = typeof(ctrl.dt)
    factor = minimum(map((c, Co) -> TF(target_courant(c, a)) / (Co + eps(TF)), ctrl.criteria, ctrl.courant))
    shrink = ctrl.explicit ? zero(TF) : TF(a.minShrink)
    set_time_step!(ctrl, ctrl.dt * clamp(factor, shrink, TF(a.maxGrow)), config)
end

# cell steps cfl/λ, Courant number ramped from startCo to maxCo; runtime.dt takes the
# turbulence step (min, mean or max of the cell steps)
function update_dt!(ctrl, a::LocalTimeStepping, model, config, iteration)
    TF = typeof(ctrl.dt)
    mesh = model.domain
    frac = a.ramp > 0 ? min(one(TF), TF(iteration - 1) / TF(a.ramp)) : one(TF)
    cfl = TF(a.startCo + (a.maxCo - a.startCo) * frac)
    rate = ctrl.rate
    @. rate = cfl / (rate + TF(1e-30))
    ctrl.dt = cfl / (ctrl.max_rate[1] + TF(1e-30))
    config.runtime.dt .= _reduce_dt(a.turbulence_dt, rate, ctrl.dt, mesh)
    return nothing
end

_reduce_dt(::MinDt, dt_cell, dt_min, mesh) = dt_min
_reduce_dt(::MaxDt, dt_cell, dt_min, mesh) = global_max(maximum(_owned(dt_cell, mesh)), mesh)
function _reduce_dt(::MeanDt, dt_cell, dt_min, mesh)
    dt = _owned(dt_cell, mesh)
    sums = global_sum([sum(dt), length(dt)], mesh)
    return sums[1] / sums[2]
end

_limit_dt!(ctrl, model, config) = _limit_dt!(ctrl, ctrl.limits, model, config)
_limit_dt!(ctrl, ::Tuple{}, model, config) = nothing
function _limit_dt!(ctrl, limits::Tuple, model, config)
    dt_max = global_min(minimum(map(l -> max_time_step(l, model, config), limits)), model.domain)
    dt_max < ctrl.dt && set_time_step!(ctrl, typeof(ctrl.dt)(dt_max), config)
    return nothing
end

set_time_step!(ctrl, dt, config) = (ctrl.dt = dt; config.runtime.dt .= dt; nothing)

# host value of runtime.dt for set-up code that runs before a TimeStepControl exists
host_dt(runtime, TF) = TF(Array(runtime.dt)[1])
