# ARCHITECTURE — motorBike import and steady CFD path, as it stands today
## the one idea
Preserve OpenFOAM's physical patch partition during import, map each patch explicitly to XCALibre boundary objects, and enforce impermeable slip through face interpolation plus consistent equation boundary contributions.
## the pipeline
| stage | file | what it owns |
|---|---|---|
| OpenFOAM token import | `src/FoamMesh/FoamMesh_1_read.jl` | points, faces, owner/neighbour, and real patch records |
| Mesh construction | `src/FoamMesh/FoamMesh_5_build.jl` | conversion from parsed data to `Mesh3` |
| Boundary mapping | `src/Discretise/Discretise_4_assign_boundaries.jl` | patch-name resolution, uniqueness, and face ranges in boundary objects |
| Slip face values | `src/Discretise/boundary_conditions/slip_interpolation.jl` | scalar extrapolation and tangential vector projection |
| Slip equation terms | `src/Discretise/boundary_conditions/slip.jl` | boundary coefficients for diffusion, convection, and sources |
| Wall equation terms | `src/Discretise/boundary_conditions/wall.jl` | fixed wall velocity diffusion and convection coefficients |
| Wall functions | `src/ModelPhysics/Turbulence/RANS_functions.jl` | aggregate incident wall faces and constrain unique wall cells |
| Pressure-flux coupling | `src/Solvers/Solvers_1_SIMPLE.jl` | shared interior and boundary pressure correction used by SIMPLE/PISO families |
| Potential-flow initialization | `src/Solvers/Solvers_0_potential_flow.jl` | pressure-derived velocity-potential projection and divergence-free initial face flux |
| Face-flux reconstruction | `src/Solvers/Solvers_5_Multiphase.jl` | all-face moment reconstruction shared by potential flow and multiphase paths |
| SIMPLE loop | `src/Solvers/Solvers_1_SIMPLE.jl` | momentum-pressure coupling, pressure relaxation, turbulence updates |
| Case definition | `/home/humberto/casesXCALibre/motorBike/motorBike.jl` | fixed turbulent `KOmega` physics, explicit numerics/runtime, native potential initialization, and patch sets |
## where each phase attaches
- Phase 1 repairs import and any evidenced boundary/setup defect, then validates the complete pipeline on the reference mesh.
- Native group representation is later work and inherits the parser's real-patch correctness.
## the invariants every phase inherits
- Boundary face ranges are contiguous, one-indexed, and cover every boundary face exactly once.
- Each field receives one boundary condition for every physical patch.
- Duplicate boundary names cannot satisfy complete patch coverage.
- Slip interpolation removes only the normal vector component and leaves scalar face values equal to their owner-cell values.
- No boundary operator invents through-wall mass flux or scalar transport.
- Prescribed velocity patches own their mass flux; pressure-adjustable patches receive deferred Extrapolated pressure flux.
- SIMPLE/PISO-family flux corrections use the full pressure solution before pressure relaxation affects cell updates.
- Nonorthogonal pressure corrections are carried into the final face mass flux in every SIMPLE/PISO-family solver.
- Face-flux reconstruction includes boundary faces and uses a scale-relative invertibility test.
- Native potential-flow initialization infers potential boundaries from pressure and returns its corrected face flux.
- A wall cell shared by multiple wall faces or patches is constrained deterministically once from an area-consistent aggregate.
- Wall-function production uses velocity relative to each wall face, including moving walls.
- CPU and accelerator-compatible code remains type-stable for configured mesh integer and float types.
## data structures worth knowing
- `FoamMeshData.boundaries` stores `Boundary{name,startFace,nFaces}` records before `Mesh3` construction.
- `Mesh3.boundaries` supplies patch names and global face ranges consumed by `assign`.
- Boundary objects carry resolved global `IDs_range` values used by interpolation and equation kernels.
## deliberate exclusions
- `inGroups` membership is discarded during import in this phase.
- OpenFOAM dictionary expansion and field-file parsing are outside this path.
