# Spellings of earlier releases: each warns once and returns the current settings

export Geometric, SmoothAggregation, AMGJacobi, AMGChebyshev, AMGGaussSeidel, AMGSOR
export AMGSymmetricSweep, AMGForwardSweep, AMGBackwardSweep, VCycle
export OnDevice, OnHost, OnDeviceKrylov, OnDeviceJacobi, OnDeviceChebyshev

_amg_deprecated(old, new) = _LOG_RANK[]() && @warn "$old is deprecated; use $new" maxlog=1 _id=Symbol(old)

struct VCycle end
struct AMGSymmetricSweep end
struct AMGForwardSweep end
struct AMGBackwardSweep end

Geometric(; merge_levels=1) = (_amg_deprecated("Geometric()", "Pairwise()"); Pairwise(merge_levels))
SmoothAggregation(; strength_threshold=0.0, smoother_weight=4/3, near_nullspace=nothing) =
    (_amg_deprecated("SmoothAggregation()", "SmoothedAggregation(threshold, weight)");
    SmoothedAggregation(threshold=strength_threshold, weight=smoother_weight, near_nullspace=near_nullspace))
AMGJacobi(; omega=4/3) = (_amg_deprecated("AMGJacobi(omega)", "Jacobi(weight)"); Jacobi(weight=omega))
AMGChebyshev(; kwargs...) = (_amg_deprecated("AMGChebyshev()", "Chebyshev()"); Chebyshev(; kwargs...))

function AMGGaussSeidel(s=nothing; sweep=AMGSymmetricSweep(), iterations=1)
    _amg_deprecated("AMGGaussSeidel()", "GaussSeidel() or SymGaussSeidel(), with AMG(sweeps=...)")
    _sweep_iterations(iterations)
    sweep = something(s, sweep)
    sweep isa AMGSymmetricSweep ? SymGaussSeidel() : _forward(sweep, GaussSeidel())
end

function AMGSOR(w=nothing, s=nothing; omega=1.0, sweep=AMGSymmetricSweep(), iterations=1)
    _amg_deprecated("AMGSOR()", "SOR(ω) or SSOR(ω), with AMG(sweeps=...)")
    _sweep_iterations(iterations)
    omega, sweep = something(w, omega), something(s, sweep)
    sweep isa AMGSymmetricSweep ? SSOR(omega) : _forward(sweep, SOR(omega))
end

_sweep_iterations(n) = n == 1 || throw(ArgumentError("smoother iterations are now AMG(sweeps=...)"))
_forward(::AMGForwardSweep, s) = s
_forward(sweep, s) = throw(ArgumentError("backward-only sweeps are no longer offered; use a forward or symmetric smoother"))

OnDevice(; max_rows=512) = (_amg_deprecated("OnDevice()", "NativeAMG(coarse=Direct())"); Direct(; max_rows))
OnHost() = (_amg_deprecated("OnHost()", "NativeAMG(coarse=Direct(max_rows=0))"); Direct(max_rows=0))
OnDeviceKrylov(; solver=Bicgstab(), rtol=1e-2, atol=0.0, itmax=50) =
    (_amg_deprecated("OnDeviceKrylov()", "NativeAMG(coarse=Iterative())"); Iterative(something(solver, Bicgstab()); rtol, atol, itmax))
OnDeviceJacobi(; omega=4/3, iterations=10) =
    (_amg_deprecated("OnDeviceJacobi()", "NativeAMG(coarse=Smoothed(Jacobi(), sweeps=n))"); Smoothed(Jacobi(weight=omega); sweeps=iterations))
OnDeviceChebyshev(; kwargs...) =
    (_amg_deprecated("OnDeviceChebyshev()", "NativeAMG(coarse=Smoothed(Chebyshev(...)))"); Smoothed(Chebyshev(; degree=10, kwargs...)))

# earlier coarse_solve values
_legacy_coarse(cs::AbstractAMGCoarseSolve) = cs
_legacy_coarse(::CPU) = Direct(max_rows=0)
_legacy_coarse(::KernelAbstractions.GPU) = Iterative()

# AMG(...) with earlier keywords: map to the current ones
function _legacy_amg(coarsening, smoother, sweeps, coarsest, refresh, native, legacy)
    kw = Dict{Symbol,Any}(pairs(legacy))
    renamed = filter(k -> k ∉ (:coarsening, :mode), collect(keys(kw)))
    isempty(renamed) || _amg_deprecated("AMG($(join(renamed, ", ")))",
        "AMG(coarsening; smoother, sweeps, coarsest, refresh, native=NativeAMG(coarse, scale_correction, fuse))")
    haskey(kw, :coarse_storage) && throw(ArgumentError("AMG(coarse_storage) is now SolverSetup(precision=MixedF32())"))
    coarsening = pop!(kw, :coarsening, coarsening)
    mode = pop!(kw, :mode, nothing)
    if haskey(kw, :pre_sweeps) || haskey(kw, :post_sweeps)
        s = Sweeps(sweeps)
        sweeps = (pop!(kw, :pre_sweeps, s.pre), pop!(kw, :post_sweeps, s.post))
    end
    refresh = pop!(kw, :coarse_refresh_interval, refresh)
    coarsest = pop!(kw, :max_coarse_rows, coarsest)
    native = NativeAMG(coarse=_legacy_coarse(pop!(kw, :coarse_solve, native.coarse)),
        scale_correction=pop!(kw, :scale_correction, native.scale_correction), fuse=pop!(kw, :fuse_levels, native.fuse))
    foreach(k -> pop!(kw, k, nothing), (:min_coarse_rows, :max_levels, :cycle))
    isempty(kw) || throw(ArgumentError("unknown AMG keywords: $(join(keys(kw), ", "))"))
    amg = AMG(coarsening; smoother, sweeps, coarsest, refresh, native)
    isnothing(mode) ? amg : _AMGMode(mode, amg)
end

# AMG(mode=...): the outer iteration of earlier releases, resolved by SolverSetup
struct _AMGMode{M,A}
    mode::M
    amg::A
end
