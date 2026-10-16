# Transient verification cases

Small transient problems that check the time discretisation: each script runs one case with
`Euler` and with `CrankNicolson` at a sequence of halved time steps, prints the error at the end
time and the observed order log₂(eᵢ/eᵢ₊₁), and leaves them in `results = (; dts, errors, orders)`.
They are verification cases (known solution or self-convergence, a measured order of accuracy)
rather than physical benchmarks, hence the folder name. The meshes are built in the scripts
with `MeshBuilder2D`; all cases run on the CPU in well under a minute each (most of it
compilation) and write no output files.

Run one with, e.g., `julia --project=<env with XCALibre> heat_decay.jl`.

| Script | Solver | What it checks | Reference | Euler | CrankNicolson |
|---|---|---|---|---|---|
| `heat_decay.jl` | `laplace!` | Time operator on a scalar; also `CrankNicolson(0.9)` (off-centred, first order) | exact semi-discrete solution 1 + cos(πx)exp(λₕt) | 1.0 | 2.0 |
| `shear_wave_piso.jl` | `piso!` | PISO stage structure (predictor, correctors) | exact semi-discrete solution sin(2πy)exp(λₕνt) | 1.0 | 2.0 |
| `shear_wave_les.jl` | `piso!`, `LES{Smagorinsky}` | time level of the eddy viscosity | CN at Δt/16 | 1.0 | 2.1 |
| `oscillating_wall.jl` | `piso!` | time-dependent boundary values (`DirichletFunction`) | CN at Δt/16 | 1.0 | 2.0 |
| `taylor_green_advected.jl` | `piso!` | convecting flux and pressure coupling | CN at Δt/16 | 0.9–1.0 | 2.0–1.8 |
| `acoustic_wave_cpiso.jl` | `cpiso!`, `Compressible`, `SensibleEnthalpy` | compressible time terms, `outer_loops` | CN at Δt/32 | see below | 2.2–2.4 |

Orders are those of the finest pairs of time steps. Notes:

- λₕ is the eigenvalue of the discrete Laplacian for the mode, so the errors of the first two cases
  are purely temporal. The other cases have no simple discrete solution and are measured against
  a run with a 16 (32) times smaller time step on the same mesh.
- `shear_wave_les.jl` uses Smagorinsky because WALE gives ν_t = 0 in pure shear.
- `taylor_green_advected.jl`: on 24² cells the CN order falls slightly at the smallest Δt; the
  part responsible shrinks with the mesh (h²Δt, from the Rhie-Chow face flux).
- `acoustic_wave_cpiso.jl`: CrankNicolson needs `outer_loops = 8` so the density of the time terms
  matches the state equation at tⁿ⁺¹; Euler runs with one pass. The Euler error of this case does
  not decrease with Δt (amplitude and phase of the wave are off by an amount independent of Δt).

Cases from the development of the Crank-Nicolson scheme that were not kept: variable time-step
patterns (they need an internal post-processing hook; `test/unit_test_crank_nicolson.jl` covers
them), the Re 100 cylinder (no order to measure), and the
3D isotropic-turbulence WALE case (a copy of `test/0_TEST_CASES/3d_incompressible_isotropic_turbulence_WALE.jl`).

`heat_decay.jl` and `shear_wave_piso.jl` are run by the test suite
(`test/unit_test_transient_verification_examples.jl`).
