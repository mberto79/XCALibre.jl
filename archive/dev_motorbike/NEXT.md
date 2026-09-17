# next: phase 1 — make the OpenFOAM motorBike case stable in XCALibre
LOAD: load the `xcalibre-dev` skill before doing anything else; it defines this file's state
  machine, the build loop and the recovery table.
updated: 2026-08-14T17:43:25+01:00
STATE: IDLE
STEP: -
HEAD: 2c17d8afc39b5e22f3e7b29d5dd0f34ef6f48a3e
BRANCH: HM/slip-boundary-fix
GIT MODE: COMMIT · PUSH: MILESTONE
GATE: julia --project=. test/runtests.jl
gate_log: ->
do: phase 1 implementation and external-case cleanup are complete
watch: preserve the unchanged validation cases and recorded 200-iteration bounds
resume: use xcalibre-close only when the user asks to close or merge the phase
## position
P1-P3 import, coverage, geometry, physical boundaries, wall aggregation and solver coupling are built.
The full repository gate is green with the original BFS test unchanged.
Uniform motorBike initialization remains unstable; the OpenFOAM potential field makes the corrected solver bounded.
P4 is gate-green: native initialization completed the sustained motorBike run and the full suite passes.
P5 fixes the external parity driver to the shipped turbulent KOmega configuration and removes investigation scaffolding.
## doing
- [x] P1 parse grouped OpenFOAM boundary dictionaries while ignoring group metadata
- [x] P2 prove exact 72-patch assignment coverage and localize the first instability
- [x] P3 correct wall/slip/symmetry, wall aggregation, moving-wall and pressure-flux behavior
- [x] P4 provide native potential-flow initialization and sustain the motorBike run
- [x] P5 remove diagnostic switches and make motorBike an explicit turbulent KOmega case
## blocked
-
## carried
- Native OpenFOAM patch-group exposure remains a separate future feature.
- FOAM3 and UNV3 share the OpenFOAM-compatible face-pyramid geometry implementation and cross-loader tests.
- The mesh writer preserves existing `constant/polyMesh` input and writes round-trippable new coordinates.
- The late motorBike hotspots are wall-adjacent, not slip-adjacent; turbulence is not the sole trigger.
- The external parity driver is `/home/humberto/casesXCALibre/motorBike/motorBike.jl`.
