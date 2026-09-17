# phase 5: distributed KOmegaSST (#1 pending)
## goal
KOmegaSST runs distributed with the same seams as KOmega. LKE/LES stay deferred.
## design
- Mirror KOmega seams into RANS_kOmegaSST.jl turbulence!:
  - wrap_eqn(k_eqn,…)/wrap_eqn(ω_eqn,…) + unwrap_eqn before update, is_distributed_mesh branch,
    same solver/petsc plumbing (solvers.k/solvers.omega).
- SST-specific risk: blending functions read CELL values on ghosts. F1/F2, CDkω, and the
  cross-diffusion dkdomegadx use k, omega, ∇k, ∇ω, ∇U at cells. Ghosts of k/omega/nut and the
  gradients MUST be consistent before these @. kernels, else blending differs across the halo.
  Audit: ensure grad!/interpolation already sync ghosts (as in serial-distributed KOmega); add
  sync! only where a cell-local @. consumes an un-synced field's ghost. Prefer reusing existing
  sync points over adding new ones (ponytail: add a sync! only if a ghost is provably stale).
- nut update + bound! are pointwise on owned; nutf interpolation needs synced nut ghosts.
## steps
- [ ] add wrap/unwrap/is_distributed seams to SST turbulence!
- [ ] trace ghost consistency for F1/F2/CDkω inputs; add minimal sync! if needed
- [ ] SST MPI gate test (mirror the KOmega distributed test)
- [ ] gate n=1,2
## gate
New SST MPI testset green n=1,2 under --project=dev/petscenv; compare a probe (e.g. nut field
norm or a residual) serial-vs-n2 within tolerance recorded in dev/baselines.json.
## risks/assumptions (concrete, from code trace)
- ∇k/∇ω via grad!(Gauss,ScalarField→VectorField) ALREADY sync ghosts (Calculate_0_gradient.jl:109)
  -> dkdomegadx/CDkω ghosts OK.
- gradU via grad!(Gauss,VectorField→TensorField) does NOT sync (line 114 variant) -> Pk/Ω ghosts
  stale -> nut ghost stale. Likely need sync!(nut) before interpolate!(nutf,nut).
- y (wall distance) wall_distance! has NO sync! -> y ghost may be stale -> arg1/arg2/F1 ghost
  wrong -> F1f wrong at boundary faces. Likely need sync!(y) once after wall_distance! (distributed).
- Seam edits: SST setup guard precon/solver with !is_distributed + wrap_eqn(label=k/omega);
  turbulence! add unwrap + k_deqn/ω_deqn + distributed||update_preconditioner + solve via *_deqn.
- Let the serial-vs-n2 nut probe DRIVE which sync!s are actually needed (add only if probe fails).
- KOmegaLKE/LES explicitly out of scope this round.
