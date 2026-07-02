# Phase 7 — AD / adjoint boundary (High)

Umbrella: `distributed_plan_detailed.md` Phase 7; `distributed_plan.md` §7 (AD strategy).
Local kernels stay AD-differentiable; MPI + linear solve get custom rules. No
differentiation through PETSc internals.

## Deliverables
1. `ChainRulesCore.rrule` for `halo_exchange!`: pullback = `halo_exchange_adjoint!`
   (ghost cotangents summed into owning cells via `unpack_add!`; kernels exist since
   Phase 2). Copy-semantics wrapper as in `distributed_plan.md` §5.
2. `psolve_transpose!` in the PETSc extension via `KSPSolveTranspose`; `rrule` for the
   distributed solve: `b̄ = solve(Aᵀ, x̄)`; `Ā = -b̄ ⊗ x` restricted to the local sparsity
   (lazy — only materialise entries present in the CSR pattern).
3. `rrule`s for `pnorm`/`pdot`/`pmean`: pullback broadcasts the (identical-on-all-ranks)
   seed to local contributions — no communication needed in the pullback beyond what the
   primal already established.
4. ChainRulesCore becomes a dep of `Distribute` (tiny, no weight concern).

## Tests (`test/distributed/test_adjoint.jl`, n = 2, 4)
- Adjoint identity `⟨v̄, Hx⟩ == ⟨Hᵀv̄, x⟩` to 1e-12 (random x, v̄; repeated).
- Gradient of a scalar loss on a small distributed diffusion case w.r.t. b and to matrix
  entries vs serial reverse-mode AD reference, 1e-6; identical across rank counts.
- FD spot-check on 1–2 design variables of a drag-like functional on the partitioned
  cavity vs serial AD gradient.

## Risks
- Enzyme/KA version pinning (known Julia 1.11 `setindex!` regressions) — pin tested
  versions in the test project; hand-written `rrule` fallback for any kernel that fails.

## Exit criteria
All identities/gradients green at n=2,4; gradients rank-count independent.
