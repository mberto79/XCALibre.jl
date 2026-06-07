# Greenfield init-once layout (plan Phase 3, prompt §2A). Built ONCE on host from the static
# mesh topology, then VRAM-resident. Maps fine cells to macro-aggregates and permutes the fine
# operator so each macro's cells/rows sit in a contiguous block — the precondition for the
# fused "one workgroup per macro-aggregate" kernel. Structure is frozen; transient coefficient
# updates only rewrite values via _amg_macro_refresh_kernel! (prompt §2B).

# Index maps stored Int32 (prompt §3: indices that fit in 32-bit halve index bandwidth). A_perm
# keeps Int CSR indices for now — Int32 packing of the operator is Phase 4 (SELL-P).
mutable struct AMGMacroLayout{VI32, MA, VV}
    fine_to_macro::VI32   # original cell -> macro id (1..n_macro)
    cell_perm::VI32       # permuted position -> original cell (new->old)
    cell_perm_inv::VI32   # original cell -> permuted position (old->new)
    agg_offsets::VI32     # macro m owns permuted positions agg_offsets[m]:agg_offsets[m+1]-1
    n_macro::Int
    A_perm::MA            # symmetric-permuted fine operator Pc·A0·Pcᵀ (AMGMatrixCSR, frozen)
    perm_value_map::VV    # A_perm nz slot -> original A0 nzval slot (value-only refresh)
end

Adapt.@adapt_structure AMGMacroLayout

# Group fine cells by macro id into contiguous blocks (counting sort), returning the permutation
# (new->old, old->new) and CSR-style macro offsets.
function _macro_permutation(fine_to_macro, n_macro)
    n = length(fine_to_macro)
    counts = zeros(Int, n_macro)
    @inbounds for i in 1:n
        counts[fine_to_macro[i]] += 1
    end
    agg_offsets = Vector{Int32}(undef, n_macro + 1)
    agg_offsets[1] = 1
    @inbounds for m in 1:n_macro
        agg_offsets[m + 1] = agg_offsets[m] + counts[m]
    end
    cell_perm = Vector{Int32}(undef, n)
    cell_perm_inv = Vector{Int32}(undef, n)
    pos = Vector{Int}(agg_offsets[1:n_macro])  # running insert position per macro
    @inbounds for i in 1:n
        m = fine_to_macro[i]
        k = pos[m]
        pos[m] += 1
        cell_perm[k] = i
        cell_perm_inv[i] = k
    end
    return cell_perm, cell_perm_inv, agg_offsets
end

# Build Pc·A0·Pcᵀ (symmetric permutation) in CSR + the slot map back to the original nzval.
# Permuted row k holds original row cell_perm[k], with columns relabelled by cell_perm_inv and
# sorted ascending. Pattern is frozen here; refresh only restreams values through perm_value_map.
# CSR indices use the narrowest signed type fitting n/nnz (Int32 on <=8GB) — the MF cycle drives
# A_perm through KA kernels, not cuSPARSE, so Int32 is safe and halves index VRAM (F1 ~36MB).
function _permuted_operator(A, cell_perm, cell_perm_inv)
    rp = _rowptr(A); cv = _colval(A); nz = _nzval(A)
    n = _m(A); nnz = length(nz); T = eltype(nz)
    TI = _amg_index_type(max(n, nnz + 1))
    new_rowptr = Vector{TI}(undef, n + 1); new_rowptr[1] = 1
    new_colval = Vector{TI}(undef, nnz)
    new_nzval = Vector{T}(undef, nnz)
    perm_value_map = Vector{Int32}(undef, nnz)
    cols = Int[]; slots = Int[]
    slot = 1
    @inbounds for k in 1:n
        old_i = cell_perm[k]
        empty!(cols); empty!(slots)
        for p in rp[old_i]:(rp[old_i + 1] - 1)
            push!(cols, cell_perm_inv[cv[p]]); push!(slots, p)
        end
        for o in sortperm(cols)
            new_colval[slot] = cols[o]
            new_nzval[slot] = nz[slots[o]]
            perm_value_map[slot] = slots[o]
            slot += 1
        end
        new_rowptr[k + 1] = slot
    end
    return AMGMatrixCSR(new_rowptr, new_colval, new_nzval, n, n), perm_value_map
end

# Build the layout once from the fine operator and the agglomeration depth (host, CPU arrays).
# Reuses the OpenFOAM faceAreaPair math (_geometric_aggregates) for the fine->macro map.
function build_macro_layout(A, merge_levels::Integer)
    Am = _amg_matrix(A)
    fine_to_macro, n_macro = _geometric_aggregates(Am, Int(merge_levels))
    n_macro >= 1 || throw(ArgumentError("macro aggregation produced no aggregates"))
    cell_perm, cell_perm_inv, agg_offsets = _macro_permutation(fine_to_macro, n_macro)
    A_perm, perm_value_map = _permuted_operator(Am, cell_perm, cell_perm_inv)
    return AMGMacroLayout(Int32.(fine_to_macro), cell_perm, cell_perm_inv, agg_offsets,
                          n_macro, A_perm, perm_value_map)
end

# Transient value-only refresh (prompt §2B): restream the new A0 coefficients into the frozen
# permuted operator. src_nzval is the ORIGINAL-ordering A0 nzval; structure never changes.
@kernel function _amg_macro_refresh_kernel!(perm_nzval, src_nzval, value_map)
    s = @index(Global)
    @inbounds perm_nzval[s] = src_nzval[value_map[s]]
end

function refresh_macro_values!(layout::AMGMacroLayout, src_nzval, backend, workgroup)
    _launch_amg_kernel!(backend, workgroup, _amg_macro_refresh_kernel!,
                        length(_nzval(layout.A_perm)), _nzval(layout.A_perm), src_nzval, layout.perm_value_map)
    return layout
end

# Validation: max|y_ref - unpermute(A_perm · permute(x))|. Zero (to round-off) confirms the
# permutation + value map reproduce the original operator action. Host loops (deterministic).
function macro_layout_spmv_error(A, layout::AMGMacroLayout)
    Am = _amg_matrix(A); n = _m(Am); T = eltype(_nzval(Am))
    x = rand(T, n)
    y_ref = _host_csr_matvec(Am, x)
    xp = Vector{T}(undef, n)
    @inbounds for k in 1:n
        xp[k] = x[layout.cell_perm[k]]
    end
    yp = _host_csr_matvec(layout.A_perm, xp)
    y_chk = Vector{T}(undef, n)
    @inbounds for k in 1:n
        y_chk[layout.cell_perm[k]] = yp[k]
    end
    return maximum(abs.(y_chk .- y_ref))
end

function _host_csr_matvec(A, x)
    rp = _rowptr(A); cv = _colval(A); nz = _nzval(A); n = _m(A)
    y = zeros(eltype(nz), n)
    @inbounds for i in 1:n, p in rp[i]:(rp[i + 1] - 1)
        y[i] += nz[p] * x[cv[p]]
    end
    return y
end
