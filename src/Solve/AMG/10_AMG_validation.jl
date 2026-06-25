# Test-only validation drivers + host oracles. Not used by the production solve/update path.

# NEW SECTION: host CSR/Jacobi primitives (deterministic oracles)

function host_csr_matvec(A, x)
    rp = _rowptr(A); cv = _colval(A); nz = _nzval(A); n = _m(A)
    y = zeros(eltype(nz), n)
    @inbounds for i in 1:n, p in rp[i]:(rp[i + 1] - 1)
        y[i] += nz[p] * x[cv[p]]
    end
    return y
end

# Weighted-Jacobi sweep matching _amg_jacobi_step_kernel! (residual form, full-row sum — P13).
function host_jacobi_sweep!(xnew, xold, b, A, invdiag, omega)
    rp = _rowptr(A); cv = _colval(A); nz = _nzval(A); T = eltype(xnew)
    @inbounds for i in 1:_m(A)
        sigma = zero(T)
        for p in rp[i]:(rp[i + 1] - 1)
            sigma += nz[p] * xold[cv[p]]
        end
        xnew[i] = xold[i] + omega * invdiag[i] * (b[i] - sigma)
    end
    return xnew
end

# NEW SECTION: independent host oracle (reference sparse P_l / A_l, natural order, plain loops)

function host_reference_vcycle(operators, Ps, omegas, invdiags, coarse_solve, b, pre, post;
                         scale_correction::Bool=false)
    L = length(operators); M = L - 1; T = eltype(b)
    xs = [zeros(T, _m(operators[l])) for l in 1:L]
    rhss = Vector{Vector{T}}(undef, L); rhss[1] = copy(b)
    for l in 1:M
        Al = operators[l]
        x = zeros(T, _m(Al)); tmp = similar(x)
        for _ in 1:pre
            host_jacobi_sweep!(tmp, x, rhss[l], Al, invdiags[l], omegas[l]); x, tmp = tmp, x
        end
        xs[l] = x
        r = rhss[l] .- host_csr_matvec(Al, x)
        rhss[l + 1] = Ps[l]' * r
    end
    xs[L] = coarse_solve(rhss[L])
    for l in M:-1:1
        Al = operators[l]
        x = xs[l]; tmp = similar(x)
        r_pre = rhss[l] .- host_csr_matvec(Al, x)           # x still holds the pre-smoothed iterate
        c = Ps[l] * xs[l + 1]
        if scale_correction
            Ac = host_csr_matvec(Al, c)
            sf = _amg_scale_factor(T(dot(r_pre, c)), T(dot(c, Ac)))
            x .+= sf .* c
        else
            x .+= c
        end
        post_rhs = l == 1 ? rhss[l] : r_pre                  # invariant 1: rhs aliasing
        for _ in 1:post
            host_jacobi_sweep!(tmp, x, post_rhs, Al, invdiags[l], omegas[l]); x, tmp = tmp, x
        end
        xs[l] = x
    end
    return xs[1]
end

# Net VRAM saved by erasing all P_l/R_l (CSR: nzval+colval+rowptr each) vs the implicit-transfer
# metadata added (row_macro n, inv_sqrt_w nc, coarse_pos nc, agg_offsets nc+1). A_l excluded (kept both ways).
function _mf_ml_vram_saved_bytes(Ps, ::Type{T}) where {T}
    erased = 0; added = 0
    for l in eachindex(Ps)
        nnzP = nnz(Ps[l]); n = size(Ps[l], 1); nc = size(Ps[l], 2)
        erased += nnzP * (sizeof(T) + sizeof(Int32)) + (n + 1) * sizeof(Int32)    # P_l (CSR)
        erased += nnzP * (sizeof(T) + sizeof(Int32)) + (nc + 1) * sizeof(Int32)   # R_l (CSR, =Pᵀ)
        added += n * sizeof(Int32) + nc * (sizeof(T) + 2 * sizeof(Int32)) + (nc + 1) * sizeof(Int32)
    end
    return erased - added
end

# NEW SECTION: matrix-free V-cycle validation drivers

# Correctness: one MF V-cycle (device, permuted) vs the host oracle (reference sparse, natural order).
function validate_matrix_free_cycle(A, merge_levels::Integer, backend; pre::Int=2, post::Int=2,
                           omega_nominal=4/3, max_coarse::Integer=64, coarse_max_rows::Integer=512,
                           scale_correction::Bool=false)
    h = build_matrix_free_hierarchy(A, merge_levels, backend; pre=pre, post=post, omega_nominal=omega_nominal,
                     max_coarse=max_coarse, coarse_max_rows=coarse_max_rows,
                     scale_correction=scale_correction)
    st = h.st; T = h.T; n = st.n
    b = rand(T, n)
    Minv_h = st.coarse_inv[] === nothing ? nothing : Array(st.coarse_inv[])  # match state's coarse mechanism
    csolve = Minv_h === nothing ? (r -> h.coarse_fac \ r) : (r -> Minv_h * r)
    x_oracle = host_reference_vcycle(h.operators, h.Ps, h.omegas, h.invdiags, csolve, b, pre, post;
                               scale_correction=scale_correction)
    b_perm = Vector{T}(undef, n)
    @inbounds for k in 1:n
        b_perm[k] = b[st.cell_perm[k]]
    end
    bd = backend isa CPU ? b_perm : Adapt.adapt(backend, b_perm)
    x_perm = Array(matrix_free_cycle!(st, bd))
    x_dev = Vector{T}(undef, n)
    @inbounds for k in 1:n
        x_dev[st.cell_perm[k]] = x_perm[k]
    end
    relerr = maximum(abs.(x_dev .- x_oracle)) / max(maximum(abs.(x_oracle)), eps(T))
    return (relerr=relerr, n=n, levels=length(st.levels) + 1, pre=pre, post=post,
            vram_saved_bytes=_mf_ml_vram_saved_bytes(h.Ps, T))
end

# Convergence (orthogonal to the oracle): MF V-cycle as a stationary preconditioner drives ‖r‖ to tol.
function validate_matrix_free_convergence(A, merge_levels::Integer, backend; pre::Int=2, post::Int=2,
                           itmax::Int=80, rtol=1e-8, omega_nominal=4/3, max_coarse::Integer=64,
                           fused_top::Integer=0, scale_correction::Bool=false)
    h = build_matrix_free_hierarchy(A, merge_levels, backend; pre=pre, post=post, omega_nominal=omega_nominal,
                     max_coarse=max_coarse, fused_top=fused_top, scale_correction=scale_correction)
    st = h.st; T = h.T; n = st.n; bk = st.backend; wg = 256
    Afine = st.levels[1].A; rp, cv, nz = _rowptr(Afine), _colval(Afine), _nzval(Afine)
    dev(v) = bk isa CPU ? copy(v) : Adapt.adapt(bk, v)
    b = dev(rand(T, n)); x = dev(zeros(T, n)); res = dev(zeros(T, n))  # finest permuted space
    _launch_amg_kernel!(bk, wg, _amg_csr_residual_kernel!, n, res, rp, cv, nz, x, b)
    KernelAbstractions.synchronize(bk)
    r0 = norm(Array(res)); rn = r0; factors = Float64[]; it = 0
    while it < itmax && rn > rtol * r0
        it += 1
        dx = matrix_free_cycle!(st, res)
        _launch_amg_kernel!(bk, wg, _amg_add_kernel!, n, x, dx)
        _launch_amg_kernel!(bk, wg, _amg_csr_residual_kernel!, n, res, rp, cv, nz, x, b)
        KernelAbstractions.synchronize(bk)
        rprev = rn; rn = norm(Array(res))
        push!(factors, rn / rprev)
    end
    return (converged=(rn <= rtol * r0), iters=it, final_rel=rn / r0,
            mean_factor=isempty(factors) ? NaN : sum(factors) / length(factors),
            last_factor=isempty(factors) ? NaN : factors[end], n=n, levels=length(st.levels) + 1)
end

# NEW SECTION: G2 zero-alloc + top-k VRAM accounting

function measure_cycle_allocations(A, merge_levels::Integer, backend; pre::Int=2, post::Int=2,
                            omega_nominal=4/3, max_coarse::Integer=64, fused_top::Integer=0,
                            coarse_max_rows::Integer=512)
    h = build_matrix_free_hierarchy(A, merge_levels, backend; pre=pre, post=post, omega_nominal=omega_nominal,
                     max_coarse=max_coarse, fused_top=fused_top, coarse_max_rows=coarse_max_rows)
    st = h.st; T = h.T; n = st.n
    rhs = backend isa CPU ? rand(T, n) : Adapt.adapt(backend, rand(T, n))
    matrix_free_cycle!(st, rhs)                                # warmup (compile + cuBLAS handle)
    a1 = @allocated matrix_free_cycle!(st, rhs)
    a2 = @allocated matrix_free_cycle!(st, rhs)                # KA launch adds a fixed host alloc — report, don't assert zero
    return (host_bytes=(a1, a2), constant=(a1 == a2), coarse_n=st.coarse_n,
            branch=(st.coarse_inv[] === nothing ? :host_lu : :gemv), levels=length(st.levels) + 1)
end

# Device bytes of the coarse operators A_l (l=2..fused_top+1) the top-k path does NOT store.
function _mf_topk_vram_erased_bytes(operators, fused_top, ::Type{T}) where {T}
    bytes = 0
    for l in 2:(fused_top + 1)
        l > length(operators) && break
        nnzA = length(_nzval(operators[l])); n = _m(operators[l])
        bytes += nnzA * (sizeof(T) + sizeof(Int32)) + (n + 1) * sizeof(Int32)
    end
    return bytes
end

# Correctness of the top-k matrix-free path: fused_top=k (Galerkin chain) vs fused_top=0 (materialized A_l).
function validate_fused_operator(A, merge_levels::Integer, backend, fused_top::Integer; pre::Int=2,
                          post::Int=2, omega_nominal=4/3, max_coarse::Integer=64)
    hk = build_matrix_free_hierarchy(A, merge_levels, backend; pre=pre, post=post, omega_nominal=omega_nominal,
                      max_coarse=max_coarse, fused_top=fused_top)
    h0 = build_matrix_free_hierarchy(A, merge_levels, backend; pre=pre, post=post, omega_nominal=omega_nominal,
                      max_coarse=max_coarse, fused_top=0)
    T = hk.T; n = hk.st.n
    b = rand(T, n)
    bd = backend isa CPU ? b : Adapt.adapt(backend, b)
    xk = Array(matrix_free_cycle!(hk.st, bd))
    x0 = Array(matrix_free_cycle!(h0.st, bd))
    relerr = maximum(abs.(xk .- x0)) / max(maximum(abs.(x0)), eps(T))
    return (relerr=relerr, n=n, levels=length(hk.st.levels) + 1, fused_top=hk.st.fused_top,
            vram_erased_bytes=_mf_topk_vram_erased_bytes(hk.operators, hk.st.fused_top, T))
end

# NEW SECTION: G1 frozen-refresh validation

# Frozen-aggregation Galerkin RAP oracle: A1's frozen aggregation/permutation applied to A2's values
# (independent host SpGEMM). A full rebuild on A2 re-aggregates differently — only the frozen oracle matches.
function host_frozen_rap_oracle(handle, A2)
    T = handle.T; M = handle.M
    op = permute_operator(_amg_matrix(A2), handle.cps[1], handle.cpis[1])[1]  # value-only perm
    ops = Any[op]; omegas = T[]
    for l in 1:M
        n = _m(ops[l]); nc = maximum(handle.aggs[l])
        iw, rm = transfer_factors(handle.aos[l], nc, n, T)
        cpos = l < M ? handle.cpis[l + 1] : Int32.(1:nc)
        cols = [Int(cpos[rm[k]]) for k in 1:n]; vals = [iw[rm[k]] for k in 1:n]
        Pperm = sparse(collect(1:n), cols, vals, n, nc)
        push!(ops, _amg_matrix(sparse(Pperm') * _csr_to_csc(ops[l]) * Pperm))
        _, invd = _diag_inverse(ops[l]); push!(omegas, _estimate_lambda_max(ops[l], invd))
    end
    return ops, omegas
end

# Validate the device refresh (A1 built -> A2 streamed, frozen structure) vs the frozen-aggregation oracle.
function validate_refresh(A1, A2, merge_levels::Integer, backend; pre::Int=2, post::Int=2,
                             omega_nominal=4/3, max_coarse::Integer=64, coarse_storage=nothing,
                             fused_top::Integer=0)
    h1 = build_matrix_free_hierarchy(A1, merge_levels, backend; pre=pre, post=post, omega_nominal=omega_nominal,
                      max_coarse=max_coarse, fused_top=fused_top, coarse_storage=coarse_storage)
    plan = build_refresh_plan(h1)
    refresh_matrix_free_hierarchy!(h1.st, plan, A2; omega_nominal=omega_nominal)
    ops, lam = host_frozen_rap_oracle(h1, A2); T = h1.T; M = length(h1.st.levels)
    relerr = 0.0; omega_relerr = 0.0
    for l in 1:M
        lv = h1.st.levels[l]
        if is_fused_level(h1.st, l)
            _, invd_ref = _diag_inverse(ops[l])                # fused level stores no values; check invdiag
            got_d = Array(lv.invdiag)
            relerr = max(relerr, maximum(abs.(got_d .- invd_ref)) / max(maximum(abs.(invd_ref)), eps(T)))
        else
            got = Array(_nzval(lv.A)); ref = _nzval(ops[l])
            length(got) == length(ref) || error("refresh/oracle pattern mismatch at level $l")
            relerr = max(relerr, maximum(abs.(got .- ref)) / max(maximum(abs.(ref)), eps(T)))
        end
        om_ref = min(T(omega_nominal), T(2) - eps(T)) / T(lam[l])
        omega_relerr = max(omega_relerr, abs(lv.omega - om_ref) / max(abs(om_ref), eps(T)))
    end
    cg = Array(_nzval(plan.coarsest_csr)); cr = _nzval(ops[end])
    relerr = max(relerr, maximum(abs.(cg .- cr)) / max(maximum(abs.(cr)), eps(T)))
    return (relerr=relerr, omega_relerr=omega_relerr, levels=M + 1, n=h1.st.n)
end

# Refreshed-state convergence: build on A1 -> refresh to A2 -> drive the stationary MF V-cycle on A2.
function validate_refresh_convergence(A1, A2, merge_levels::Integer, backend; pre::Int=2, post::Int=2,
                                   itmax::Int=200, rtol=1e-8, omega_nominal=4/3, max_coarse::Integer=64,
                                   coarse_storage=nothing, fused_top::Integer=0)
    h1 = build_matrix_free_hierarchy(A1, merge_levels, backend; pre=pre, post=post, omega_nominal=omega_nominal,
                      max_coarse=max_coarse, fused_top=fused_top, coarse_storage=coarse_storage)
    plan = build_refresh_plan(h1)
    refresh_matrix_free_hierarchy!(h1.st, plan, A2; omega_nominal=omega_nominal)
    st = h1.st; T = h1.T; n = st.n; bk = backend; wg = 256
    Afine = st.levels[1].A; rp, cv, nz = _rowptr(Afine), _colval(Afine), _nzval(Afine)
    dev(v) = bk isa CPU ? copy(v) : Adapt.adapt(bk, v)
    b = dev(rand(T, n)); x = dev(zeros(T, n)); res = dev(zeros(T, n))
    _launch_amg_kernel!(bk, wg, _amg_csr_residual_kernel!, n, res, rp, cv, nz, x, b)
    KernelAbstractions.synchronize(bk)
    r0 = norm(Array(res)); rn = r0; it = 0
    while it < itmax && rn > rtol * r0
        it += 1
        dx = matrix_free_cycle!(st, res)
        _launch_amg_kernel!(bk, wg, _amg_add_kernel!, n, x, dx)
        _launch_amg_kernel!(bk, wg, _amg_csr_residual_kernel!, n, res, rp, cv, nz, x, b)
        KernelAbstractions.synchronize(bk)
        rn = norm(Array(res))
    end
    return (converged=(rn <= rtol * r0), iters=it, final_rel=rn / r0, n=n, levels=length(st.levels) + 1)
end

# NEW SECTION: end-to-end path comparison (REPL driver)

# Run `mode` through update!+_amg_solve_mode! on both the matrix-free and reference paths, same device
# system. A and b must already be on `backend`. n_update>1 exercises the transient refresh path.
function compare_amg_paths(A, b, backend; mode=Cg(), merge_levels=1, fuse_levels=1,
                                coarse_storage=Float64, itmax=200, rtol=1e-8, atol=1e-8,
                                workgroup=64, scale_correction=true, n_update=1)
    config = (hardware=(backend=backend, workgroup=workgroup),)
    run_one = (solver) -> begin
        ws = _workspace(solver, b)
        for _ in 1:n_update; update!(ws, A, solver, config); end
        x = similar(b); fill!(x, 0)
        _amg_solve_mode!(ws, ws.hierarchy, solver, solver.mode, A, b, x; itmax=itmax, atol=atol, rtol=rtol)
        (iters=ws.iterations, converged=ws.converged, relres=ws.last_relative_residual,
         factor=ws.hierarchy.last_cycle_factor, matrix_free=(ws.hierarchy isa MatrixFreeHierarchy))
    end
    gf_solver = AMG(mode=mode, coarsening=Geometric(merge_levels=merge_levels), smoother=AMGJacobi(),
                    fuse_levels=fuse_levels, coarse_storage=coarse_storage, max_coarse_rows=4096,
                    scale_correction=scale_correction)
    ref_solver = AMG(mode=mode, coarsening=Geometric(merge_levels=merge_levels), smoother=AMGJacobi(),
                     fuse_levels=0, max_coarse_rows=4096)
    return (matrix_free=run_one(gf_solver), reference=run_one(ref_solver))
end
