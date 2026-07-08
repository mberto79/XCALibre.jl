# Phase 2 — Halo exchange + distributed fields (Medium-High)

Umbrella: `distributed_plan_detailed.md` §2.4, §5 of `distributed_plan.md`.
Depends on Phase 1's ProcessorPatch ordering invariant (send/recv lists align by
construction — no schedule negotiation needed).

## Files
- `src/Distribute/Distribute_2_halo.jl` — HaloExchange build + kernels + exchange.
- `src/Distribute/Distribute_3_fields.jl` — field wrappers, `sync!`, global reductions.

## HaloExchange
```julia
struct HaloExchange{VI,VB}
    comm::MPI.Comm
    patches::Vector{ProcessorPatch{VI}}   # from DistributedMesh.procs
    send_bufs::Vector{VB}; recv_bufs::Vector{VB}   # backend arrays, width 1 (scalar) or 3
    send_reqs::Vector{MPI.Request}; recv_reqs::Vector{MPI.Request}
    cuda_aware::Bool
end
```
- Built once per width (1 and 3) from `dmesh.procs`; buffers via
  `KernelAbstractions.allocate(backend, TF, width * n)`. Reused every call.
- Kernels: `pack!`, `unpack!`, `unpack_add!` (Atomix; adjoint use in Phase 7) — exactly as
  in `distributed_plan.md` §5. Vector fields pack x,y,z into ONE 3-wide buffer per
  neighbour (one message, not three).
- `halo_exchange!(vals, H, backend, workgroup)`: post all `Irecv!` first → `pack!` →
  `KernelAbstractions.synchronize` → `Isend` → `Waitall(recv)` → `unpack!` → sync →
  `Waitall(send)`. Tag = 0 (one exchange in flight at a time by construction).
- `halo_exchange_adjoint!`: reverse scatter — pack ghost cotangents, send owner-ward,
  `unpack_add!` into `send_cells`. Written now, exercised in Phase 7.
- Host staging: when `!MPI.has_cuda()` and backend is GPU, mirror pinned host buffers,
  device→host after pack, host→device before unpack. CPU backend: device==host buffers.
  Per Q6 there is NO compute fallback — staging only moves message buffers.

## Fields & reductions
```julia
struct DistributedScalarField{F<:ScalarField,H}  field::F; halo::H end
struct DistributedVectorField{F<:VectorField,H}  field::F; halo::H end
sync!(df, config)          # one halo exchange of df.field values (3-wide for vector)
```
- Overloads (on existing generics, per §2.6 budget): `initialise!` (delegate + sync).
- New: `pnorm`, `pdot`, `pmean` — computed over OWNED entries only (`1:n_owned`) +
  `MPI.Allreduce(+)`; ghost entries must never enter reductions.

## Tests (`test/distributed/test_halo.jl`, mpiexec n = 2, 4; CPU)
- Exchange cell centres: each ghost's received value equals neighbour's owned centre bitwise.
- Linear field f(x)=a·x+b: ghosts match analytic exactly (1-layer halo sufficiency).
- Vector exchange via `VectorField` (x,y,z all correct through the 3-wide buffer).
- `pnorm`/`pdot` == serial `norm`/`dot` on gathered field, 1e-14.
- Idempotency: two consecutive exchanges give identical state (buffer reuse correct).
- Adjoint identity smoke: `⟨v̄, Hx⟩ == ⟨Hᵀv̄, x⟩` to 1e-12 (full battery in Phase 7).

## Exit criteria
All tests green under the Phase 0 MPI harness at n=2 and n=4; no allocations in
`halo_exchange!` after warmup (buffers reused) — check with `@allocated`.
