# OpenFOAM motorBike parity — design specification
## goal
XCALibre loads and runs the supplied OpenFOAM 12 motorBike mesh and broadly equivalent steady incompressible k-omega setup without numerical explosion.
The diagnosis distinguishes mesh parsing, patch assignment, boundary discretisation, and solver configuration using reproducible evidence.
## vocabulary
- Patch group: OpenFOAM `inGroups` metadata naming several patches; it is not itself a boundary patch in this phase.
- Slip: impermeable, shear-free boundary; vector face values have zero normal component and scalar quantities use zero normal gradient.
- Stable: all evolved fields remain finite and physically bounded while residuals do not show sustained exponential growth.
- Reference case: `/home/humberto/casesOpenFOAM/motorBikeSteady` using OpenFOAM 12.
- XCALibre case: `/home/humberto/casesXCALibre/motorBike` using this repository branch.
## requirements
- R1 `FOAM3D_mesh` shall load boundary files containing one-line or multiline `inGroups` entries while ignoring group metadata.
- R2 Mesh import shall preserve the declared patch count, patch names, face counts, and one-indexed start faces exactly.
- R3 The XCALibre case shall assign every physical mesh patch exactly once for each solved or derived field, without relying on native group support.
- R4 Slip velocity treatment shall enforce zero normal face velocity/flux and no tangential viscous traction consistently with OpenFOAM 12; slip scalar treatment shall be zero-gradient.
- R5 Any code defect found in the instability path shall have a focused regression test that fails on the defective behavior.
- R6 The corrected case shall complete a sustained steady run with finite bounded U, p, k, omega, and nut and no missing boundary assignments.
- R7 Existing supported mesh imports and boundary-condition cases shall remain regression-green.
- R8 The XCALibre motorBike driver shall be a self-contained turbulent `KOmega` case with explicit runtime, relaxation, initialization, and native potential-flow settings rather than environment-controlled diagnostic modes.
## acceptance
- The supplied grouped boundary file imports all 72 real patches and their ranges match the file declarations.
- A programmatic coverage check reports no missing or duplicate patch assignments for U, p, k, omega, or nut.
- Focused tests demonstrate slip projection and operator contributions for representative oblique normals and flux directions.
- The corrected motorBike case passes a smoke run before a sustained run; reported extrema and residual history identify the stability outcome.
- The repository's prescribed full test command passes, or any unrelated/pre-existing failure is isolated and reported with evidence.
## deliberately NOT requirements
- Patch groups are not exposed as Julia mesh groups or accepted directly in `assign`.
- Bitwise or iteration-for-iteration agreement with OpenFOAM is not required.
- Changes to the OpenFOAM reference case are not required.
## interpretation flags
- A sustained run means at least the existing 200-iteration evidence horizon unless convergence or a stronger bounded horizon is reached sooner.
- Broadly equivalent permits solver/preconditioner differences but not materially different physical boundary conditions.
