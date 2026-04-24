using XCALibre
using Adapt
using LinearAlgebra
using Printf
using Serialization

const REPLAY_DRIFT_INTERVAL = 5

function make_cylinder_probe_case(iterations::Integer)
    grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
    mesh_file = joinpath(grids_dir, "cylinder_d10mm_5mm.unv")
    mesh = UNV2D_mesh(mesh_file, scale=0.001)

    backend = CPU()
    workgroup = 1024
    activate_multithread(backend)
    hardware = Hardware(backend=backend, workgroup=workgroup)
    mesh_dev = adapt(backend, mesh)

    velocity = [0.5, 0.0, 0.0]
    noSlip = [0.0, 0.0, 0.0]
    nu = 1e-3

    model = Physics(
        time=Transient(),
        fluid=Fluid{Incompressible}(nu=nu),
        turbulence=RANS{Laminar}(),
        energy=Energy{Isothermal}(),
        domain=mesh_dev
    )

    BCs = assign(
        region=mesh_dev,
        (
            U = [
                Dirichlet(:inlet, velocity),
                Zerogradient(:outlet),
                Wall(:cylinder, noSlip),
                Extrapolated(:bottom),
                Extrapolated(:top)
            ],
            p = [
                Zerogradient(:inlet),
                Dirichlet(:outlet, 0.0),
                Wall(:cylinder),
                Extrapolated(:bottom),
                Extrapolated(:top)
            ]
        )
    )

    solvers = (
        U = SolverSetup(
            solver=Bicgstab(),
            preconditioner=Jacobi(),
            convergence=1e-7,
            relax=1.0,
            rtol=0.0,
            atol=1e-5
        ),
        p = SolverSetup(
            solver=AMG(
                mode=:cg,
                cycle=:V,
                coarsening=SmoothAggregation(),
                smoother=AMGJacobi()
            ),
            preconditioner=Jacobi(),
            convergence=1e-7,
            relax=1.0,
            itmax=40,
            rtol=0.0,
            atol=1e-5
        )
    )

    schemes = (
        U = Schemes(time=CrankNicolson, divergence=LUST, gradient=Gauss),
        p = Schemes(time=CrankNicolson, gradient=Gauss)
    )

    runtime = Runtime(iterations=iterations, write_interval=-1, time_step=0.0025)
    config = Configuration(solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

    initialise!(model.momentum.U, velocity)
    initialise!(model.momentum.p, 0.0)
    return model, config
end

function capture_pressure_systems(path::AbstractString; samples=3)
    reset_pressure_matrix_captures!()
    enable_pressure_matrix_capture!(limit=samples)
    model, config = make_cylinder_probe_case(samples)
    run!(model, config)
    disable_pressure_matrix_capture!()

    systems = pressure_matrix_captures()
    open(path, "w") do io
        serialize(io, systems)
    end
    @printf("CAPTURED systems=%d path=%s\n", length(systems), path)
    return systems
end

function load_pressure_systems(path::AbstractString)
    open(path, "r") do io
        return deserialize(io)
    end
end

function probe_configs(profile::String)
    if profile == "full"
        strengths = (0.08, 0.12, 0.18, 0.25)
        smoother_weights = (0.7, 1.0, 4 / 3)
        jacobi_omegas = (2 / 3, 0.8, 1.1)
        sweep_pairs = ((1, 1), (2, 2))
        coarse_rows = (128, 256, 512, 1024)
        truncations = (0, 4, 5)
    elseif profile == "near_default"
        strengths = (0.14, 0.16, 0.18, 0.20)
        smoother_weights = (1.0,)
        jacobi_omegas = (1.1,)
        sweep_pairs = ((2, 2),)
        coarse_rows = (128, 256, 512)
        truncations = (4,)
    elseif profile == "sa_quality"
        strengths = (0.08, 0.12, 0.16, 0.20, 0.24, 0.30)
        smoother_weights = (0.6, 0.8, 1.0, 1.2, 1.4)
        jacobi_omegas = (1.1,)
        sweep_pairs = ((2, 2),)
        coarse_rows = (512,)
        truncations = (2, 3, 4, 5, 6)
    elseif profile == "cycle_budget"
        strengths = (0.16,)
        smoother_weights = (1.0,)
        jacobi_omegas = (1.1,)
        sweep_pairs = ((1, 1), (2, 2), (3, 3), (4, 4))
        coarse_rows = (128, 256, 512, 1024, 2048)
        truncations = (4,)
    elseif profile == "jacobi_damping"
        strengths = (0.16,)
        smoother_weights = (1.0,)
        jacobi_omegas = (0.45, 0.55, 0.60, 2 / 3, 0.75, 0.85, 1.0)
        sweep_pairs = ((2, 2),)
        coarse_rows = (512,)
        truncations = (4,)
    elseif profile == "jacobi_high"
        strengths = (0.16,)
        smoother_weights = (1.0,)
        jacobi_omegas = (0.90, 1.0, 1.10, 1.20, 4 / 3)
        sweep_pairs = ((2, 2),)
        coarse_rows = (512,)
        truncations = (4,)
    elseif profile == "complexity_controlled"
        strengths = (0.10, 0.14, 0.18)
        smoother_weights = (0.8, 1.0)
        jacobi_omegas = (0.8, 1.1)
        sweep_pairs = ((1, 1), (2, 2))
        coarse_rows = (128, 256, 512)
        truncations = (2, 3, 4)
    else
        strengths = (0.12, 0.18, 0.25)
        smoother_weights = (0.7, 1.0)
        jacobi_omegas = (2 / 3, 1.1)
        sweep_pairs = ((1, 1), (2, 2))
        coarse_rows = (128, 512)
        truncations = (0, 4)
    end

    configs = NamedTuple[]
    for strength in strengths, smoother_weight in smoother_weights, omega in jacobi_omegas,
        sweeps in sweep_pairs, max_coarse_rows in coarse_rows, max_entries in truncations
        push!(configs, (
            strength_threshold=strength,
            smoother_weight=smoother_weight,
            jacobi_omega=omega,
            presweeps=sweeps[1],
            postsweeps=sweeps[2],
            max_coarse_rows=max_coarse_rows,
            max_prolongation_entries=max_entries,
            level_strength_thresholds=profile == "complexity_controlled" ? (strength, max(0.04, strength * 0.75), max(0.02, strength * 0.5)) : nothing,
            aggressive_levels=profile == "complexity_controlled" ? 1 : 0,
            aggressive_passes=1,
            coarse_drop_tolerances=profile == "complexity_controlled" ? (0.0, 0.01, 0.03, 0.05) : ()
        ))
    end
    return configs
end

function cfg_value(cfg, key::Symbol, default)
    return key in keys(cfg) ? getproperty(cfg, key) : default
end

function level_rows(hierarchy)
    return join(map(level -> string(size(level.A, 1)), hierarchy.levels), "->")
end

function first_coarse_rows(hierarchy)
    return length(hierarchy.levels) >= 2 ? size(hierarchy.levels[2].A, 1) : size(hierarchy.levels[1].A, 1)
end

function make_probe_solver(cfg; coarse_refresh_interval=typemax(Int), numeric_refresh_rtol=Inf)
    return AMG(
        mode=:cg,
        cycle=:V,
        coarsening=SmoothAggregation(
            strength_threshold=cfg.strength_threshold,
            level_strength_thresholds=cfg_value(cfg, :level_strength_thresholds, nothing),
            smoother_weight=cfg.smoother_weight,
            max_prolongation_entries=cfg.max_prolongation_entries,
            aggressive_levels=cfg_value(cfg, :aggressive_levels, 0),
            aggressive_passes=cfg_value(cfg, :aggressive_passes, 1),
            coarse_drop_tolerances=cfg_value(cfg, :coarse_drop_tolerances, ())
        ),
        smoother=AMGJacobi(omega=cfg.jacobi_omega),
        presweeps=cfg.presweeps,
        postsweeps=cfg.postsweeps,
        max_coarse_rows=cfg.max_coarse_rows,
        coarse_refresh_interval=coarse_refresh_interval,
        numeric_refresh_rtol=numeric_refresh_rtol
    )
end

function run_probe(system, cfg; itmax=40, atol=1e-5, rtol=0.0)
    solver = make_probe_solver(cfg)
    b = copy(system.b)
    x = copy(system.x0)
    workspace = XCALibre.Solve._workspace(solver, b)
    hierarchy = XCALibre.Solve.setup_hierarchy(system.A, solver, CPU(); log_diagnostics=false)
    workspace.hierarchy = hierarchy
    vcycle_factor = vcycle_reduction_factor!(workspace, hierarchy, solver, system.A, b, x)
    elapsed_s = @elapsed XCALibre.Solve.amg_cg_solve!(
        workspace,
        hierarchy,
        solver,
        system.A,
        b,
        x;
        itmax=itmax,
        atol=atol,
        rtol=rtol
    )
    final_residual = isempty(workspace.residual_history) ? NaN : workspace.residual_history[end]
    apply_calls = workspace.timing.apply_calls
    return (
        iterations=workspace.iterations,
        apply_calls=apply_calls,
        avg_vcycle_time_s=apply_calls > 0 ? workspace.timing.apply_time_s / apply_calls : 0.0,
        solve_time_s=elapsed_s,
        operator_complexity=hierarchy.operator_complexity,
        grid_complexity=hierarchy.grid_complexity,
        vcycle_factor=vcycle_factor,
        final_residual=final_residual,
        final_relative_residual=workspace.last_relative_residual,
        first_coarse_rows=first_coarse_rows(hierarchy),
        rows=level_rows(hierarchy)
    )
end

function run_history_probe(system, cfg; itmax=120, atol=1e-5, rtol=0.0)
    solver = make_probe_solver(cfg)
    b = copy(system.b)
    x = copy(system.x0)
    workspace = XCALibre.Solve._workspace(solver, b)
    hierarchy = XCALibre.Solve.setup_hierarchy(system.A, solver, CPU(); log_diagnostics=false)
    workspace.hierarchy = hierarchy
    vcycle_factor = vcycle_reduction_factor!(workspace, hierarchy, solver, system.A, b, x)
    elapsed_s = @elapsed XCALibre.Solve.amg_cg_solve!(
        workspace,
        hierarchy,
        solver,
        system.A,
        b,
        x;
        itmax=itmax,
        atol=atol,
        rtol=rtol
    )
    bnorm = max(norm(b), eps(eltype(b)))
    relative_history = workspace.residual_history ./ bnorm
    return (
        iterations=workspace.iterations,
        solve_time_s=elapsed_s,
        avg_vcycle_time_s=workspace.timing.apply_calls > 0 ? workspace.timing.apply_time_s / workspace.timing.apply_calls : 0.0,
        apply_calls=workspace.timing.apply_calls,
        vcycle_factor=vcycle_factor,
        final_relative_residual=workspace.last_relative_residual,
        relative_history=relative_history,
        rows=level_rows(hierarchy),
        operator_complexity=hierarchy.operator_complexity,
        grid_complexity=hierarchy.grid_complexity
    )
end

function reuse_policies(profile::String)
    if profile in ("reuse", "near_default", "smoke", "full")
        return [
            (name="finest_only", coarse_refresh_interval=typemax(Int), numeric_refresh_rtol=Inf),
            (name="full_every_system", coarse_refresh_interval=typemax(Int), numeric_refresh_rtol=0.0),
            (name="full_every_2_systems", coarse_refresh_interval=1, numeric_refresh_rtol=Inf),
            (name="full_every_3_systems", coarse_refresh_interval=2, numeric_refresh_rtol=Inf),
            (name="rtol_0p01", coarse_refresh_interval=typemax(Int), numeric_refresh_rtol=0.01),
            (name="rtol_0p03", coarse_refresh_interval=typemax(Int), numeric_refresh_rtol=0.03),
            (name="rtol_0p05", coarse_refresh_interval=typemax(Int), numeric_refresh_rtol=0.05),
            (name="rtol_0p10", coarse_refresh_interval=typemax(Int), numeric_refresh_rtol=0.10)
        ]
    end
    throw(ArgumentError("unknown replay policy profile: $profile"))
end

function default_replay_config()
    return (
        strength_threshold=0.16,
        smoother_weight=1.0,
        jacobi_omega=1.1,
        presweeps=2,
        postsweeps=2,
        max_coarse_rows=512,
        max_prolongation_entries=4,
        level_strength_thresholds=nothing,
        aggressive_levels=0,
        aggressive_passes=1,
        coarse_drop_tolerances=()
    )
end

function _diagnostic_range(values)
    isempty(values) && return (NaN, NaN)
    return (minimum(values), maximum(values))
end

function _near_zero_threshold(T)
    return sqrt(eps(float(T)))
end

function amg_cg_solve_diagnostics!(workspace, hierarchy, solver, A, b, x; itmax, atol, rtol, drift_interval=REPLAY_DRIFT_INTERVAL)
    T = eltype(x)
    r = workspace.residual
    z = workspace.preconditioned
    p = workspace.search
    q = workspace.q
    dotpq_values = Float64[]
    rz_values = Float64[]
    drift_values = Float64[]
    nonpositive_dotpq = false
    nonpositive_rz = false
    nearzero_dotpq = false
    nearzero_rz = false
    threshold = _near_zero_threshold(T)

    XCALibre.Solve._residual!(r, A, x, b)
    XCALibre.Solve._reset_residual_history!(workspace)
    bnorm = max(norm(b), eps(T))
    rnorm = norm(r)
    XCALibre.Solve._push_residual_norm_history!(workspace, rnorm)
    rel = rnorm / bnorm
    initial_rel = rel
    if rnorm <= atol || rel <= rtol
        workspace.iterations = 0
        workspace.last_relative_residual = rel
        XCALibre.Solve._update_cycle_factor!(hierarchy, initial_rel, rel, 0, solver)
        return (
            dotpq_min=NaN,
            dotpq_max=NaN,
            rz_min=NaN,
            rz_max=NaN,
            nonpositive_dotpq=false,
            nonpositive_rz=false,
            nearzero_dotpq=false,
            nearzero_rz=false,
            max_recurrence_drift=0.0
        )
    end

    elapsed_s = @elapsed XCALibre.Solve.amg_apply_preconditioner!(z, hierarchy, solver, r)
    XCALibre.Solve._record_apply_timing!(workspace, elapsed_s)
    copyto!(p, z)
    rz = dot(r, z)
    push!(rz_values, float(rz))
    nonpositive_rz |= rz <= zero(T)
    nearzero_rz |= abs(rz) <= threshold

    k = 0
    while k < itmax && rnorm > atol && rel > rtol
        k += 1
        mul!(q, A, p)
        dotpq = dot(p, q)
        push!(dotpq_values, float(dotpq))
        nonpositive_dotpq |= dotpq <= zero(T)
        nearzero_dotpq |= abs(dotpq) <= threshold
        if dotpq <= zero(T) || abs(dotpq) <= eps(T)
            break
        end
        alpha = rz / dotpq
        @inbounds for i in eachindex(x)
            x[i] += alpha * p[i]
            r[i] -= alpha * q[i]
        end
        rnorm = norm(r)
        if drift_interval > 0 && k % drift_interval == 0
            true_rnorm = norm(XCALibre.Solve._residual!(workspace.correction, A, x, b))
            push!(drift_values, abs(true_rnorm - rnorm) / max(true_rnorm, eps(T)))
        end
        XCALibre.Solve._push_residual_norm_history!(workspace, rnorm)
        rel = rnorm / bnorm
        if rnorm <= atol || rel <= rtol
            break
        end
        elapsed_s = @elapsed XCALibre.Solve.amg_apply_preconditioner!(z, hierarchy, solver, r)
        XCALibre.Solve._record_apply_timing!(workspace, elapsed_s)
        rz_new = dot(r, z)
        push!(rz_values, float(rz_new))
        nonpositive_rz |= rz_new <= zero(T)
        nearzero_rz |= abs(rz_new) <= threshold
        if rz <= zero(T) || abs(rz) <= eps(T)
            break
        end
        beta = rz_new / rz
        @inbounds for i in eachindex(p)
            p[i] = z[i] + beta * p[i]
        end
        rz = rz_new
    end
    workspace.iterations = k
    workspace.last_relative_residual = rel
    XCALibre.Solve._update_cycle_factor!(hierarchy, initial_rel, rel, k, solver)
    dotpq_min, dotpq_max = _diagnostic_range(dotpq_values)
    rz_min, rz_max = _diagnostic_range(rz_values)
    return (
        dotpq_min=dotpq_min,
        dotpq_max=dotpq_max,
        rz_min=rz_min,
        rz_max=rz_max,
        nonpositive_dotpq=nonpositive_dotpq,
        nonpositive_rz=nonpositive_rz,
        nearzero_dotpq=nearzero_dotpq,
        nearzero_rz=nearzero_rz,
        max_recurrence_drift=isempty(drift_values) ? 0.0 : maximum(drift_values)
    )
end

function vcycle_reduction_factor!(workspace, hierarchy, solver, A, b, x0)
    r = workspace.residual
    z = workspace.correction
    XCALibre.Solve._residual!(r, A, x0, b)
    r0 = norm(r)
    r0 <= eps(eltype(r)) && return 0.0
    elapsed_s = @elapsed XCALibre.Solve.amg_apply_preconditioner!(z, hierarchy, solver, r)
    XCALibre.Solve._record_apply_timing!(workspace, elapsed_s)
    mul!(workspace.q, A, z)
    @inbounds for i in eachindex(r)
        r[i] -= workspace.q[i]
    end
    return norm(r) / r0
end

function run_reuse_policy(systems, cfg, policy; itmax=40, atol=1e-5, rtol=0.0)
    solver = make_probe_solver(
        cfg;
        coarse_refresh_interval=policy.coarse_refresh_interval,
        numeric_refresh_rtol=policy.numeric_refresh_rtol
    )
    b0 = copy(first(systems).b)
    workspace = XCALibre.Solve._workspace(solver, b0)
    config = (hardware=Hardware(backend=CPU(), workgroup=1024),)
    residuals = Float64[]
    vcycle_factors = Float64[]
    rows_by_system = String[]
    iterations = Int[]
    solve_time_s = 0.0
    update_time_s = 0.0
    build_before = 0.0
    refresh_before = 0.0
    finest_refresh_before = 0.0
    apply_before = 0.0
    apply_calls_before = 0
    dotpq_min = Inf
    dotpq_max = -Inf
    rz_min = Inf
    rz_max = -Inf
    nonpositive_dotpq = false
    nonpositive_rz = false
    nearzero_dotpq = false
    nearzero_rz = false
    max_drift = 0.0

    for (system_id, system) in enumerate(systems)
        b = copy(system.b)
        x = copy(system.x0)
        update_time_s += @elapsed XCALibre.Solve.update!(workspace, system.A, solver, config)
        hierarchy = workspace.hierarchy
        push!(vcycle_factors, vcycle_reduction_factor!(workspace, hierarchy, solver, system.A, b, x))
        diag = nothing
        elapsed_s = @elapsed begin
            diag = amg_cg_solve_diagnostics!(
                workspace,
                hierarchy,
                solver,
                system.A,
                b,
                x;
                itmax=itmax,
                atol=atol,
                rtol=rtol
            )
        end
        solve_time_s += elapsed_s
        push!(residuals, workspace.last_relative_residual)
        push!(rows_by_system, level_rows(hierarchy))
        push!(iterations, workspace.iterations)
        dotpq_min = min(dotpq_min, diag.dotpq_min)
        dotpq_max = max(dotpq_max, diag.dotpq_max)
        rz_min = min(rz_min, diag.rz_min)
        rz_max = max(rz_max, diag.rz_max)
        nonpositive_dotpq |= diag.nonpositive_dotpq
        nonpositive_rz |= diag.nonpositive_rz
        nearzero_dotpq |= diag.nearzero_dotpq
        nearzero_rz |= diag.nearzero_rz
        max_drift = max(max_drift, diag.max_recurrence_drift)
        @printf(
            "REPLAY system=%d policy=%s update_action=%s iterations=%d final_relative=%.6e vcycle_factor=%.6e rows=%s\n",
            system_id,
            policy.name,
            string(workspace.timing.last_update_action),
            workspace.iterations,
            workspace.last_relative_residual,
            vcycle_factors[end],
            rows_by_system[end]
        )
    end

    timing = workspace.timing
    build_time_s = timing.build_time_s - build_before
    refresh_time_s = timing.refresh_time_s - refresh_before
    finest_refresh_time_s = timing.finest_refresh_time_s - finest_refresh_before
    apply_time_s = timing.apply_time_s - apply_before
    apply_calls = timing.apply_calls - apply_calls_before
    hierarchy = workspace.hierarchy
    return (
        policy=policy,
        systems=length(systems),
        iterations=iterations,
        final_relative_residual=last(residuals),
        max_relative_residual=maximum(residuals),
        mean_relative_residual=sum(residuals) / length(residuals),
        residuals=residuals,
        solve_time_s=solve_time_s,
        update_time_s=update_time_s,
        avg_vcycle_time_s=apply_calls > 0 ? apply_time_s / apply_calls : 0.0,
        build_time_s=build_time_s,
        build_calls=timing.build_calls,
        refresh_time_s=refresh_time_s,
        refresh_calls=timing.refresh_calls,
        finest_refresh_time_s=finest_refresh_time_s,
        finest_refresh_calls=timing.finest_refresh_calls,
        apply_calls=apply_calls,
        operator_complexity=hierarchy.operator_complexity,
        grid_complexity=hierarchy.grid_complexity,
        rows=level_rows(hierarchy),
        rows_by_system=rows_by_system,
        vcycle_factors=vcycle_factors,
        dotpq_min=isfinite(dotpq_min) ? dotpq_min : NaN,
        dotpq_max=isfinite(dotpq_max) ? dotpq_max : NaN,
        rz_min=isfinite(rz_min) ? rz_min : NaN,
        rz_max=isfinite(rz_max) ? rz_max : NaN,
        nonpositive_dotpq=nonpositive_dotpq,
        nonpositive_rz=nonpositive_rz,
        nearzero_dotpq=nearzero_dotpq,
        nearzero_rz=nearzero_rz,
        max_recurrence_drift=max_drift
    )
end

function _result_metric(row, metric::Symbol)
    metric == :max_relative_residual && return row.result.max_relative_residual
    metric == :solve_time_s && return row.result.solve_time_s
    metric == :avg_vcycle_time_s && return row.result.avg_vcycle_time_s
    metric == :refresh_time_s && return row.result.refresh_time_s
    metric == :operator_complexity && return row.result.operator_complexity
    metric == :composite && return row.composite_score
    throw(ArgumentError("unknown replay ranking metric: $metric"))
end

function _add_replay_composite_scores(rows)
    isempty(rows) && return rows
    metrics = (:max_relative_residual, :solve_time_s, :avg_vcycle_time_s, :refresh_time_s, :operator_complexity)
    weights = Dict(
        :max_relative_residual => 0.45,
        :solve_time_s => 0.25,
        :avg_vcycle_time_s => 0.10,
        :refresh_time_s => 0.10,
        :operator_complexity => 0.10
    )
    bounds = Dict{Symbol,Tuple{Float64,Float64}}()
    for metric in metrics
        values = Float64[_result_metric(row, metric) for row in rows if isfinite(_result_metric(row, metric))]
        bounds[metric] = isempty(values) ? (0.0, 0.0) : (minimum(values), maximum(values))
    end
    scored = NamedTuple[]
    for row in rows
        composite_score = 0.0
        for metric in metrics
            best, worst = bounds[metric]
            composite_score += weights[metric] * _normalise_metric(_result_metric(row, metric), best, worst)
        end
        push!(scored, merge(row, (composite_score=composite_score,)))
    end
    return scored
end

function _format_vector(values)
    return join(map(value -> @sprintf("%.6e", value), values), ",")
end

function print_replay_rankings(rows; top_n=8)
    scored = _add_replay_composite_scores(rows)
    isempty(scored) && return nothing
    for metric in (:max_relative_residual, :solve_time_s, :avg_vcycle_time_s, :refresh_time_s, :operator_complexity, :composite)
        sorted = sort(scored, by=row -> _result_metric(row, metric))
        @printf("TOP_REPLAY metric=%s count=%d\n", string(metric), min(top_n, length(sorted)))
        for (rank, row) in enumerate(Iterators.take(sorted, top_n))
            result = row.result
            policy = result.policy
            @printf(
                "TOP_REPLAY metric=%s rank=%d policy=%s coarse_interval=%s numeric_rtol=%s final_relative=%.6e max_relative=%.6e mean_relative=%.6e solve_s=%.6e avg_vcycle_s=%.6e build_s=%.6e refresh_s=%.6e refreshes=%d finest_refreshes=%d operator_complexity=%.6f composite=%.6f rows=%s residuals=%s vcycle_factors=%s dotpq_min=%.6e dotpq_max=%.6e rz_min=%.6e rz_max=%.6e nonpositive_dotpq=%s nonpositive_rz=%s nearzero_dotpq=%s nearzero_rz=%s max_drift=%.6e\n",
                string(metric),
                rank,
                policy.name,
                policy.coarse_refresh_interval == typemax(Int) ? "Inf" : string(policy.coarse_refresh_interval),
                isinf(policy.numeric_refresh_rtol) ? "Inf" : @sprintf("%.3f", policy.numeric_refresh_rtol),
                result.final_relative_residual,
                result.max_relative_residual,
                result.mean_relative_residual,
                result.solve_time_s,
                result.avg_vcycle_time_s,
                result.build_time_s,
                result.refresh_time_s,
                result.refresh_calls,
                result.finest_refresh_calls,
                result.operator_complexity,
                row.composite_score,
                result.rows,
                _format_vector(result.residuals),
                _format_vector(result.vcycle_factors),
                result.dotpq_min,
                result.dotpq_max,
                result.rz_min,
                result.rz_max,
                string(result.nonpositive_dotpq),
                string(result.nonpositive_rz),
                string(result.nearzero_dotpq),
                string(result.nearzero_rz),
                result.max_recurrence_drift
            )
        end
    end
    return scored
end

function run_reuse_replay(systems, policies)
    cfg = default_replay_config()
    rows = NamedTuple[]
    for (policy_id, policy) in enumerate(policies)
        result = run_reuse_policy(systems, cfg, policy)
        push!(rows, (policy_id=policy_id, result=result))
        @printf(
            "PROBE_REPLAY policy=%s coarse_interval=%s numeric_rtol=%s systems=%d final_relative=%.6e max_relative=%.6e mean_relative=%.6e solve_s=%.6e avg_vcycle_s=%.6e build_s=%.6e refresh_s=%.6e refreshes=%d finest_refreshes=%d operator_complexity=%.6f grid_complexity=%.6f composite_pending rows=%s residuals=%s vcycle_factors=%s dotpq_min=%.6e dotpq_max=%.6e rz_min=%.6e rz_max=%.6e nonpositive_dotpq=%s nonpositive_rz=%s nearzero_dotpq=%s nearzero_rz=%s max_drift=%.6e\n",
            policy.name,
            policy.coarse_refresh_interval == typemax(Int) ? "Inf" : string(policy.coarse_refresh_interval),
            isinf(policy.numeric_refresh_rtol) ? "Inf" : @sprintf("%.3f", policy.numeric_refresh_rtol),
            result.systems,
            result.final_relative_residual,
            result.max_relative_residual,
            result.mean_relative_residual,
            result.solve_time_s,
            result.avg_vcycle_time_s,
            result.build_time_s,
            result.refresh_time_s,
            result.refresh_calls,
            result.finest_refresh_calls,
            result.operator_complexity,
            result.grid_complexity,
            result.rows,
            _format_vector(result.residuals),
            _format_vector(result.vcycle_factors),
            result.dotpq_min,
            result.dotpq_max,
            result.rz_min,
            result.rz_max,
            string(result.nonpositive_dotpq),
            string(result.nonpositive_rz),
            string(result.nearzero_dotpq),
            string(result.nearzero_rz),
            result.max_recurrence_drift
        )
    end
    return print_replay_rankings(rows)
end

function _metric_value(row, metric::Symbol)
    metric == :final_relative_residual && return row.result.final_relative_residual
    metric == :solve_time_s && return row.result.solve_time_s
    metric == :avg_vcycle_time_s && return row.result.avg_vcycle_time_s
    metric == :operator_complexity && return row.result.operator_complexity
    metric == :vcycle_factor && return row.result.vcycle_factor
    metric == :composite && return row.composite_score
    throw(ArgumentError("unknown probe ranking metric: $metric"))
end

function _config_metric_value(row, metric::Symbol)
    metric == :max_relative_residual && return row.max_relative_residual
    metric == :mean_relative_residual && return row.mean_relative_residual
    metric == :solve_time_s && return row.solve_time_s
    metric == :avg_vcycle_time_s && return row.avg_vcycle_time_s
    metric == :mean_vcycle_factor && return row.mean_vcycle_factor
    metric == :max_vcycle_factor && return row.max_vcycle_factor
    metric == :operator_complexity && return row.operator_complexity
    metric == :grid_complexity && return row.grid_complexity
    metric == :first_coarse_rows && return row.first_coarse_rows
    metric == :composite && return row.composite_score
    throw(ArgumentError("unknown config ranking metric: $metric"))
end

function _normalise_metric(value, best, worst)
    span = worst - best
    isfinite(value) || return Inf
    span <= eps(Float64) && return 0.0
    return (value - best) / span
end

function _add_composite_scores(rows)
    isempty(rows) && return rows
    metrics = (:final_relative_residual, :solve_time_s, :avg_vcycle_time_s, :operator_complexity, :vcycle_factor)
    bounds = Dict{Symbol,Tuple{Float64,Float64}}()
    for metric in metrics
        values = Float64[_metric_value(row, metric) for row in rows if isfinite(_metric_value(row, metric))]
        bounds[metric] = isempty(values) ? (0.0, 0.0) : (minimum(values), maximum(values))
    end
    scored = NamedTuple[]
    for row in rows
        residual_best, residual_worst = bounds[:final_relative_residual]
        solve_best, solve_worst = bounds[:solve_time_s]
        vcycle_best, vcycle_worst = bounds[:avg_vcycle_time_s]
        op_best, op_worst = bounds[:operator_complexity]
        factor_best, factor_worst = bounds[:vcycle_factor]
        composite_score =
            0.45 * _normalise_metric(row.result.final_relative_residual, residual_best, residual_worst) +
            0.20 * _normalise_metric(row.result.solve_time_s, solve_best, solve_worst) +
            0.15 * _normalise_metric(row.result.avg_vcycle_time_s, vcycle_best, vcycle_worst) +
            0.10 * _normalise_metric(row.result.operator_complexity, op_best, op_worst) +
            0.10 * _normalise_metric(row.result.vcycle_factor, factor_best, factor_worst)
        push!(scored, merge(row, (composite_score=composite_score,)))
    end
    return scored
end

function _add_config_composite_scores(rows)
    isempty(rows) && return rows
    metrics = (:max_relative_residual, :solve_time_s, :avg_vcycle_time_s, :mean_vcycle_factor, :operator_complexity)
    weights = Dict(
        :max_relative_residual => 0.45,
        :solve_time_s => 0.20,
        :avg_vcycle_time_s => 0.10,
        :mean_vcycle_factor => 0.15,
        :operator_complexity => 0.10
    )
    bounds = Dict{Symbol,Tuple{Float64,Float64}}()
    for metric in metrics
        values = Float64[_config_metric_value(row, metric) for row in rows if isfinite(_config_metric_value(row, metric))]
        bounds[metric] = isempty(values) ? (0.0, 0.0) : (minimum(values), maximum(values))
    end
    scored = NamedTuple[]
    for row in rows
        composite_score = 0.0
        for metric in metrics
            best, worst = bounds[metric]
            composite_score += weights[metric] * _normalise_metric(_config_metric_value(row, metric), best, worst)
        end
        push!(scored, merge(row, (composite_score=composite_score,)))
    end
    return scored
end

function print_probe_rankings(rows; top_n=8)
    scored = _add_composite_scores(rows)
    isempty(scored) && return nothing
    for metric in (:final_relative_residual, :vcycle_factor, :solve_time_s, :avg_vcycle_time_s, :operator_complexity, :composite)
        sorted = sort(scored, by=row -> _metric_value(row, metric))
        @printf("TOP metric=%s count=%d\n", string(metric), min(top_n, length(sorted)))
        for (rank, row) in enumerate(Iterators.take(sorted, top_n))
            cfg = row.cfg
            result = row.result
            @printf(
                "TOP metric=%s rank=%d system=%d config=%d strength=%.3f sa_weight=%.3f jacobi_omega=%.3f sweeps=%d/%d max_coarse_rows=%d max_p_entries=%d final_relative=%.6e vcycle_factor=%.6e solve_s=%.6e avg_vcycle_s=%.6e operator_complexity=%.6f grid_complexity=%.6f first_coarse_rows=%d composite=%.6f rows=%s\n",
                string(metric),
                rank,
                row.system_id,
                row.config_id,
                cfg.strength_threshold,
                cfg.smoother_weight,
                cfg.jacobi_omega,
                cfg.presweeps,
                cfg.postsweeps,
                cfg.max_coarse_rows,
                cfg.max_prolongation_entries,
                result.final_relative_residual,
                result.vcycle_factor,
                result.solve_time_s,
                result.avg_vcycle_time_s,
                result.operator_complexity,
                result.grid_complexity,
                result.first_coarse_rows,
                row.composite_score,
                result.rows
            )
        end
    end
    return nothing
end

function summarise_config_rows(rows)
    by_config = Dict{Int,Vector{NamedTuple}}()
    for row in rows
        push!(get!(by_config, row.config_id, NamedTuple[]), row)
    end

    summary = NamedTuple[]
    for config_id in sort(collect(keys(by_config)))
        config_rows = by_config[config_id]
        cfg = first(config_rows).cfg
        residuals = [row.result.final_relative_residual for row in config_rows]
        vcycle_factors = [row.result.vcycle_factor for row in config_rows]
        solve_times = [row.result.solve_time_s for row in config_rows]
        avg_vcycle_times = [row.result.avg_vcycle_time_s for row in config_rows]
        operator_complexities = [row.result.operator_complexity for row in config_rows]
        grid_complexities = [row.result.grid_complexity for row in config_rows]
        first_rows = [row.result.first_coarse_rows for row in config_rows]
        push!(summary, (
            config_id=config_id,
            cfg=cfg,
            systems=length(config_rows),
            final_relative_residual=last(residuals),
            max_relative_residual=maximum(residuals),
            mean_relative_residual=sum(residuals) / length(residuals),
            residuals=residuals,
            mean_vcycle_factor=sum(vcycle_factors) / length(vcycle_factors),
            max_vcycle_factor=maximum(vcycle_factors),
            vcycle_factors=vcycle_factors,
            solve_time_s=sum(solve_times),
            avg_vcycle_time_s=sum(avg_vcycle_times) / length(avg_vcycle_times),
            operator_complexity=sum(operator_complexities) / length(operator_complexities),
            grid_complexity=sum(grid_complexities) / length(grid_complexities),
            first_coarse_rows=round(Int, sum(first_rows) / length(first_rows)),
            first_coarse_rows_by_system=first_rows,
            rows_by_system=[row.result.rows for row in config_rows]
        ))
    end
    return summary
end

function print_config_rankings(rows; top_n=12)
    scored = _add_config_composite_scores(rows)
    isempty(scored) && return nothing
    for metric in (:max_relative_residual, :mean_vcycle_factor, :solve_time_s, :avg_vcycle_time_s, :operator_complexity, :first_coarse_rows, :composite)
        sorted = sort(scored, by=row -> _config_metric_value(row, metric))
        @printf("TOP_CONFIG metric=%s count=%d\n", string(metric), min(top_n, length(sorted)))
        for (rank, row) in enumerate(Iterators.take(sorted, top_n))
            cfg = row.cfg
            @printf(
                "TOP_CONFIG metric=%s rank=%d config=%d systems=%d strength=%.3f sa_weight=%.3f jacobi_omega=%.3f sweeps=%d/%d max_coarse_rows=%d max_p_entries=%d final_relative=%.6e max_relative=%.6e mean_relative=%.6e mean_vcycle_factor=%.6e max_vcycle_factor=%.6e solve_s=%.6e avg_vcycle_s=%.6e operator_complexity=%.6f grid_complexity=%.6f first_coarse_rows=%d composite=%.6f residuals=%s vcycle_factors=%s rows=%s\n",
                string(metric),
                rank,
                row.config_id,
                row.systems,
                cfg.strength_threshold,
                cfg.smoother_weight,
                cfg.jacobi_omega,
                cfg.presweeps,
                cfg.postsweeps,
                cfg.max_coarse_rows,
                cfg.max_prolongation_entries,
                row.final_relative_residual,
                row.max_relative_residual,
                row.mean_relative_residual,
                row.mean_vcycle_factor,
                row.max_vcycle_factor,
                row.solve_time_s,
                row.avg_vcycle_time_s,
                row.operator_complexity,
                row.grid_complexity,
                row.first_coarse_rows,
                row.composite_score,
                _format_vector(row.residuals),
                _format_vector(row.vcycle_factors),
                join(row.rows_by_system, "|")
            )
        end
    end
    return scored
end

function run_probe_sweep(systems, configs)
    rows = NamedTuple[]
    for (system_id, system) in enumerate(systems)
        for (config_id, cfg) in enumerate(configs)
            result = run_probe(system, cfg)
            push!(rows, (
                system_id=system_id,
                config_id=config_id,
                cfg=cfg,
                result=result
            ))
            @printf(
                "PROBE system=%d config=%d strength=%.3f sa_weight=%.3f jacobi_omega=%.3f sweeps=%d/%d max_coarse_rows=%d max_p_entries=%d iterations=%d apply_calls=%d vcycle_factor=%.6e avg_vcycle_s=%.6e solve_s=%.6e operator_complexity=%.6f grid_complexity=%.6f first_coarse_rows=%d final_residual=%.6e final_relative=%.6e rows=%s\n",
                system_id,
                config_id,
                cfg.strength_threshold,
                cfg.smoother_weight,
                cfg.jacobi_omega,
                cfg.presweeps,
                cfg.postsweeps,
                cfg.max_coarse_rows,
                cfg.max_prolongation_entries,
                result.iterations,
                result.apply_calls,
                result.vcycle_factor,
                result.avg_vcycle_time_s,
                result.solve_time_s,
                result.operator_complexity,
                result.grid_complexity,
                result.first_coarse_rows,
                result.final_residual,
                result.final_relative_residual,
                result.rows
            )
        end
    end
    print_probe_rankings(rows)
    print_config_rankings(summarise_config_rows(rows))
    return rows
end

function _history_sample(history)
    points = unique!(sort!([1, 2, 3, 6, 11, 21, 41, 81, length(history)]))
    return join(map(i -> @sprintf("%d:%.6e", i - 1, history[i]), points), ",")
end

function run_history_sweep(systems, configs)
    rows = NamedTuple[]
    for (system_id, system) in enumerate(systems)
        for (config_id, cfg) in enumerate(configs)
            result = run_history_probe(system, cfg)
            push!(rows, (
                system_id=system_id,
                config_id=config_id,
                cfg=cfg,
                result=result
            ))
            @printf(
                "HISTORY system=%d config=%d strength=%.3f sa_weight=%.3f jacobi_omega=%.3f sweeps=%d/%d max_coarse_rows=%d max_p_entries=%d iterations=%d apply_calls=%d final_relative=%.6e solve_s=%.6e avg_vcycle_s=%.6e operator_complexity=%.6f grid_complexity=%.6f rows=%s rel_history=%s\n",
                system_id,
                config_id,
                cfg.strength_threshold,
                cfg.smoother_weight,
                cfg.jacobi_omega,
                cfg.presweeps,
                cfg.postsweeps,
                cfg.max_coarse_rows,
                cfg.max_prolongation_entries,
                result.iterations,
                result.apply_calls,
                result.final_relative_residual,
                result.solve_time_s,
                result.avg_vcycle_time_s,
                result.operator_complexity,
                result.grid_complexity,
                result.rows,
                _history_sample(result.relative_history)
            )
        end
    end
    scored = _add_composite_scores(rows)
    for metric in (:final_relative_residual, :solve_time_s, :avg_vcycle_time_s, :operator_complexity, :composite)
        sorted = sort(scored, by=row -> _metric_value(row, metric))
        @printf("TOP_HISTORY metric=%s count=%d\n", string(metric), min(8, length(sorted)))
        for (rank, row) in enumerate(Iterators.take(sorted, 8))
            cfg = row.cfg
            result = row.result
            @printf(
                "TOP_HISTORY metric=%s rank=%d system=%d config=%d sweeps=%d/%d max_coarse_rows=%d final_relative=%.6e solve_s=%.6e avg_vcycle_s=%.6e operator_complexity=%.6f composite=%.6f rows=%s rel_history=%s\n",
                string(metric),
                rank,
                row.system_id,
                row.config_id,
                cfg.presweeps,
                cfg.postsweeps,
                cfg.max_coarse_rows,
                result.final_relative_residual,
                result.solve_time_s,
                result.avg_vcycle_time_s,
                result.operator_complexity,
                row.composite_score,
                result.rows,
                _history_sample(result.relative_history)
            )
        end
    end
    return rows
end

function main(args)
    action = get(args, 1, "capture_sweep")
    samples = length(args) >= 2 ? parse(Int, args[2]) : 3
    path = get(args, 3, joinpath(@__DIR__, "amg_pressure_probe_systems.bin"))
    profile = get(args, 4, "smoke")

    systems = if action == "capture" || action == "capture_sweep"
        capture_pressure_systems(path; samples=samples)
    else
        load_pressure_systems(path)
    end

    if action == "sweep" || action == "capture_sweep"
        run_probe_sweep(systems, probe_configs(profile))
    elseif action == "history"
        run_history_sweep(systems, probe_configs(profile))
    elseif action == "replay"
        run_reuse_replay(systems, reuse_policies(profile))
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
