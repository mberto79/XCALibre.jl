# phase 1: motorBike parity and stability
## goal
Make the supplied OpenFOAM 12 mesh importable without manual boundary-file edits and make the equivalent XCALibre steady k-omega case remain bounded through a sustained run.
## design
- Parse boundary dictionaries structurally: create a patch only for a token preceding `{`, accept `nFaces` and `startFace` only inside that patch, and ignore all other entries including nested/list-valued `inGroups` metadata.
- Derive expected patch coverage from the imported mesh, compare names as sets with multiplicity for every field, and make the Julia case build wall/slip/inlet/outlet sets from that evidence.
- Compare slip and no-slip face evaluation plus matrix/source coefficients against OpenFOAM's constraint fields and XCALibre's operator sign conventions before changing formulas.
- Aggregate incident wall-function faces to unique wall cells deterministically before writing production values or constraining equation rows.
- Reproduce with short, logged runs and field extrema; change setup or code only where an isolated test demonstrates the cause.
## steps
- [x] P1 replace the boundary line heuristic with group-tolerant real-patch parsing; add compact one-line and multiline group fixtures asserting exact names/counts/ranges and malformed-count errors.
- [x] P2a enforce exact patch-name multiplicity in assignment and prove that the real case covers all 72 imported patches.
- [x] P2b localize the first saved extrema to cell 210839 and its three wall faces, compare that cell with the bounded OpenFOAM result, and exclude slip adjacency as the initiating topology.
- [x] P3a make Wall/RotatingWall full-vector fixed-value diffusion constraints and make Slip vector diffusion match OpenFOAM basicSymmetry projection.
- [x] P3b area-average production and omega over all incident wall-function faces and update/constrain each unique wall cell once.
- [x] P3c correct moving-ground, backflow, BoundedUpwind and pressure-relaxation coupling; isolate potential-flow initialization as the remaining stability requirement (D8-D10).
- [x] P4 add native potential-flow projection, all-face reconstruction, and nonorthogonal flux bookkeeping; sustain motorBike and pass the full gate (D11-D12).
- [x] P5 remove environment-controlled diagnostic modes and GPU-incompatible investigation reports; leave the external motorBike driver as an explicit turbulent `KOmega` case (D13).
## gate
- `julia --project=. test/test_mesh_conversion.jl`
- Focused slip/operator test selected or added during P3.
- Focused oblique no-slip and multi-face/multi-patch wall-cell aggregation tests selected or added during P3.
- `julia --project=. test/runtests.jl`
- MotorBike smoke run of 2 iterations followed by at least 200 iterations with finite quantitatively bounded fields, complete patch coverage, and deterministic 1-thread/multithread behavior.
## risks/assumptions
- Existing output directories are diagnostic evidence only and may have been produced before the current boundary file or uncommitted edits.
- A 354k-cell full suite plus motorBike run is affordable locally; long commands write logs under ignored `dev/logs/`.
- OpenFOAM and XCALibre use the byte-identical mesh, so geometry differences cannot explain divergent behavior after successful import.
- Outlet `Zerogradient` is not identical to OpenFOAM `inletOutlet`; backflow behavior must be measured before attributing instability to slip.
- The first observed extrema are at cell 210839, incident to three motorbike wall faces and no slip faces; its equation row and wall-function aggregation are the primary defect probes.
