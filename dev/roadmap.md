# roadmap: XCALibre.jl distributed (MPI) module

## now

Active phase and its exit gate: `dev/phaseRoadmap.md`. Current step and state: `dev/activeContext.md`.

## requirements

`dev/spec.md` is authoritative. P1 serves R1 through R12 (R11, R12 added by D72).

## phases

- [x] P0 - build the distributed module over eight phases, then a five-phase round of configuration, launch, logging, HYPRE and SST work | archived, `archive/dev_distributed/`
- [ ] P1 - release polish: make the module stock-installable, environment-variable free, rank-uniform by construction, scaling-attributed, gated and documented, then structurally ready to scale (D72) | `dev/phaseRoadmap.md`
- [ ] P2 - HPC validation: multi-node CPU, multi-GPU and AMD runs of the M18-M24 code, scaling recorded against R7/Q3, then M22-S6 (overlap) and M23-S4 (parallel repartition) if either was withdrawn locally (D73) | opens when P1 closes

## flagged

- [P1] `archive/dev_distributed/` and `archive/dev_motorbike/` are the only copies of the pre-P1 record; `dev/` was gitignored until P1-M1 (D1).
- [P1] Distributed LKE and LES remain unwired and inherit the SST synchronisation audit; spec lists them as documented gaps, not defects.
- [P1] `dev/archive/reviews/p1/audit-2026-09-18.md` (2026-09-18) is the source of M18-M24; archived at M18 close (D77). Its GPU evidence is one device and two ranks; P2 owns everything beyond that.
