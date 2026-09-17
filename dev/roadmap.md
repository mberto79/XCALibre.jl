# roadmap: XCALibre.jl distributed (MPI) module

## now

Active phase and its exit gate: `dev/phaseRoadmap.md`. Current step and state: `dev/activeContext.md`.

## requirements

`dev/spec.md` is authoritative. P1 serves R1 through R10.

## phases

- [x] P0 - build the distributed module over eight phases, then a five-phase round of configuration, launch, logging, HYPRE and SST work | archived, `archive/dev_distributed/`
- [ ] P1 - release polish: make the module stock-installable, environment-variable free, rank-uniform by construction, scaling-attributed, gated and documented | `dev/phaseRoadmap.md`

## flagged

- [P1] `archive/dev_distributed/` and `archive/dev_motorbike/` are the only copies of the pre-P1 record; `dev/` was gitignored until P1-M1 (D1).
- [P1] Distributed LKE and LES remain unwired and inherit the SST synchronisation audit; spec lists them as documented gaps, not defects.
- [P1] `wall_distance!` reports a spurious non-convergence; it is a pre-existing serial artefact of comparing a residual to a fixed 1e-15 threshold.
