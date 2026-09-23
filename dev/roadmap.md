# roadmap: XCALibre.jl distributed (MPI) module

## now

Active phase and its exit gate: `dev/phaseRoadmap.md`. Current step and state: `dev/activeContext.md`.

## requirements

`dev/spec.md` is authoritative. P1 serves R1 through R14 (R11, R12 added by D72; R13, R14 by D161).

## phases

- [x] P0 - build the distributed module over eight phases, then a five-phase round of configuration, launch, logging, HYPRE and SST work | archived, `archive/dev_distributed/`
- [ ] P1 - release polish: make the module stock-installable, environment-variable free, rank-uniform by construction, scaling-attributed, gated and documented, then structurally ready to scale (D72), then moving fewer bytes per iteration on every backend (M28-M31, D159), progress switch (M32) and MPI-path audit (M33); all milestones closed, exit gate pending | `dev/phaseRoadmap.md`
- [ ] P2 - HPC validation: multi-node CPU, multi-GPU and AMD runs of the M18-M24 code, scaling recorded against R7/Q3, then M22-S6 (overlap), M22-S8 (owned-row kernels at n=64) and M23-S4 (parallel repartition) if withdrawn locally (D73, D115, D117) | opens when P1 closes

## flagged

- [P1] `archive/dev_distributed/` and `archive/dev_motorbike/` are the only copies of the pre-P1 record; `dev/` was gitignored until P1-M1 (D1).
- [P1] Distributed LKE and LES remain unwired and inherit the SST synchronisation audit; spec lists them as documented gaps, not defects.
- [P1] Open findings to triage at P1 close: VTK `initialise_writer` builds output strings (~3 s on motorBike) even with `write_interval=-1`, threaded path only (D202); stock PETSc_jll 3.25.4 has only Int64-index libraries, Int32 would be −7.6% at n=4 but the user keeps the stock default (D205).
- [P1] `dev/archive/reviews/p1/audit-2026-09-18.md` (2026-09-18) is the source of M18-M24; archived at M18 close (D77). Its GPU evidence is one device and two ranks; P2 owns everything beyond that.
