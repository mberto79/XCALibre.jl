# Distributed (MPI) simulations

XCALibre.jl can run a simulation across several MPI ranks (multi-core, multi-node or multi-GPU),
using PETSc for the distributed linear solves. The physics setup, boundary conditions, schemes and
`run!` call are the same as in a serial script. Only the mesh is partitioned and distributed, so
you do not write a solver, model or boundary condition any differently for parallel execution.

## Requirements

You need Julia 1.10 or later and `PETSc` and `MPI` in your project environment, and nothing else.
The binaries that `PETSc_jll` and `MPI.jl` install are enough for a Float64 CPU run, with no
preferences file and no shell configuration. Two cases need more:

- `BoomerAMG()` needs a PETSc built with hypre. The stock `Float64` libraries include hypre, so it
  works out of the box at the default precision. `Float32` builds do not include it.
- GPU runs need an NVIDIA GPU and a CUDA-enabled PETSc, which `PETSc_jll` does not ship. The
  matrix, right-hand side and solution then stay on the device. Without a CUDA-enabled PETSc, a
  GPU run stops with an error; it is never moved onto the host. A prebuilt CUDA-enabled PETSc can
  be installed without compiling, see [GPU runs without compiling PETSc](@ref).

To use a system MPI or your own PETSc build, see [Setting up MPI and PETSc](@ref).

## Setting up MPI and PETSc

This section is only for users who need something the stock binaries do not provide: the MPI
library installed on a cluster, a PETSc built with particular options, or a GPU-enabled PETSc.

### How the pieces fit

- `MPI.jl` chooses the MPI library, set with the `MPIPreferences` package. `PETSc_jll`
  follows that choice and loads its matching build.
- `PETSc.jl` can load any PETSc library you point it at instead of `PETSc_jll`. That library
  must be built against the same MPI that `MPI.jl` uses.
- Both choices are stored in the `LocalPreferences.toml` file of the project environment and are
  read when packages precompile. Restart Julia after changing either one.
- The scalar precision and the library path are fixed by the environment you run in and cannot be
  switched at run time. Use a separate project environment for each PETSc build.

### Using a system MPI

In the project environment, point `MPI.jl` at the library and its launcher, then restart Julia
and instantiate again. When the new library has a different ABI (for example, Open MPI instead of
MPICH), the instantiate step downloads the matching `PETSc_jll` artifacts. Without it,
`using PETSc` fails with a missing-artifact error.

```julia
using MPIPreferences
MPIPreferences.use_system_binary(; library_names=["/path/to/lib/libmpi"],
    mpiexec="/path/to/bin/mpiexec")
# restart Julia, then:
using Pkg; Pkg.instantiate()
using MPI; MPI.versioninfo()   # confirms which library is loaded
```

`MPIPreferences.use_jll_binary("OpenMPI_jll")` (or `"MPICH_jll"`, `"MPItrampoline_jll"`)
switches back to a Julia-provided MPI. On clusters, `MPItrampoline` lets one Julia environment use
the site MPI through a small wrapper library; see the
[MPI.jl configuration guide](https://juliaparallel.org/MPI.jl/stable/configuration/). Details
are not repeated here.

### Pointing PETSc.jl at your own PETSc

```julia
using PETSc
PETSc.set_library!("/path/to/lib/libpetsc.so"; PetscScalar=Float64, PetscInt=Int32)
# restart Julia, then precompile before the first mpiexec launch:
using Pkg; Pkg.precompile()
PETSc.library_info()   # shows the configured library, scalar and index types
```

- `PetscInt` must match the library: `Int64` if it was configured with `--with-64-bit-indices`
  (`PETSC_USE_64BIT_INDICES` in `petscconf.h`), `Int32` otherwise.
- Precompile in a plain Julia session before launching with `mpiexec`. Precompiling inside the
  ranks can fail with "Precompiled image ... not available with flags".
- `PETSc.unset_library!()` returns the environment to `PETSc_jll`.

### GPU runs without compiling PETSc

conda-forge publishes CUDA-enabled PETSc builds (`cuda12_real_*` and `cuda13_real_*`, Float64 with
32-bit indices, for Linux x86-64 and aarch64). Each build comes with a matching MPI in the same
conda environment, and `MPI.jl` and `PETSc.jl` are pointed at both. The steps below were verified
with PETSc 3.25.5 (`cuda12_real`) on an NVIDIA RTX 4070 at one and two ranks, with Open MPI and
with MPICH. [micromamba](https://mamba.readthedocs.io/) installs it without root access. The
environment takes about 3 GB.

```bash
micromamba create -n petsc-cuda -c conda-forge 'petsc=*=cuda12_real*' openmpi
```

Pick `cuda13_real` instead if your driver supports CUDA 13 (`nvidia-smi` shows the highest CUDA
version the driver supports). Then, in a fresh Julia project environment holding XCALibre,
`MPI`, `PETSc` and `CUDA`, run the two setup blocks above with the paths below. Restart Julia
after each block.

```julia
prefix = "/path/to/micromamba/envs/petsc-cuda"   # the conda environment's prefix
MPIPreferences.use_system_binary(; library_names=["$prefix/lib/libmpi"],
    mpiexec="$prefix/bin/mpiexec")
# restart, Pkg.instantiate(), then:
PETSc.set_library!("$prefix/lib/libpetsc.so"; PetscScalar=Float64, PetscInt=Int32)
# restart, Pkg.precompile()
```

Both MPI variants work without further settings. The MPI variant decides how messages between
ranks travel; the solves run on the GPU either way:

- **Open MPI** from conda-forge is built with CUDA support, but it is off until you export
  `OMPI_MCA_opal_cuda_support=true` before launching. With it on, device buffers pass directly
  between ranks.
- **MPICH** from conda-forge is not CUDA-aware, so messages are staged through host memory.

Launch with `mpiexecjl` as usual; it uses the conda environment's `mpiexec`. The conda-forge
hypre runs on the host only, so use `GAMG()` or `Jacobi()` for pressure on the GPU. `BoomerAMG()`
with GPU fields crashes on this build instead of running. Float32 GPU runs are not available from
conda-forge and still need a compiled PETSc.

### Compiling a CUDA-enabled PETSc

Build PETSc against the MPI that `MPI.jl` uses, then select both as above:

```bash
./configure --prefix=$HOME/petsc-cuda --with-debugging=0 --with-shared-libraries=1 \
  --with-mpi-dir=$MPI_DIR --with-cuda=1 --with-cuda-arch=<sm, e.g. 89> \
  --download-hypre --download-fblaslapack \
  COPTFLAGS=-O3 CXXOPTFLAGS=-O3 FOPTFLAGS=-O3 CUDAOPTFLAGS=-O3
make all install
```

`--download-hypre` with CUDA builds a GPU-capable hypre, so `BoomerAMG()` runs on the device (see
[On the GPU](@ref)). Add `--with-precision=single` for a Float32 build, in its own project
environment. Spack (`spack install petsc+cuda`) and the E4S containers are alternatives.
The PETSc.jl [HPC notes](https://github.com/JuliaParallel/PETSc.jl/blob/main/docs/src/man/hpc.md)
cover cluster builds.

### GPU communication between ranks

At start-up XCALibre asks the MPI library whether it is CUDA-aware (`MPI.has_cuda()`) and prints
which path it takes, once, from rank 0:

- CUDA-aware: device buffers go straight to MPI, for XCALibre's halo exchange and for PETSc.
- Not CUDA-aware: a warning says that messages between ranks are staged through host memory. The
  solves still run on the GPU; only the exchanged boundary values take the extra copies, which
  cost more as ranks and interface sizes grow.

An explicit `-use_gpu_aware_mpi` in `petsc_options` overrides PETSc's choice.

### What a CUDA-aware MPI alone gives

Nothing for the linear solves. PETSc's CUDA support is fixed when PETSc is compiled, and
`PETSc_jll` has none in any variant. A CUDA-aware MPI with the stock `PETSc_jll` can exchange
device buffers, but XCALibre still refuses a GPU run because the solves would have to move to the
host. A GPU run needs a CUDA-enabled PETSc; a CUDA-aware MPI then removes the host staging of
messages.

### Troubleshooting

- *"Artifact ... was not found"* when loading PETSc after switching MPI: run `Pkg.instantiate()`
  in the environment.
- *"Precompiled image ... not available with flags"* under `mpiexec`: run `Pkg.precompile()` in a
  plain session first.
- *"MPI is not CUDA-aware"* warning with an MPI you expect to be CUDA-aware: check that its CUDA
  support is enabled at launch (Open MPI: `OMPI_MCA_opal_cuda_support=true`).
- *"fields live on the GPU but this PETSc build has no cuda support"*: the environment is still
  using `PETSc_jll`; check `PETSc.library_info()`.
- Ranks abort at start-up in `MPI_Init`: the `mpiexec` on your `PATH` belongs to a different MPI.
  Launch through `mpiexecjl`, which always uses the configured one.

## Distributing the mesh

Wrap the mesh read in [`distribute`](@ref). Every rank makes the same call. The reader runs only on
rank 0, and the partitioned mesh is scattered so that each rank gets its own `DistributedMesh`.
You do not need to call `MPI.Init()` or write a `rank == 0` guard:

```jldoctest distributed; filter = r".*"s => s"", output = false
using XCALibre, PETSc

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh_file = joinpath(grids_dir, "backwardFacingStep_10mm.unv")

mesh = distribute() do
    UNV2D_mesh(mesh_file, scale=0.001)
end

# output

```

!!! warning
    Do not assign the mesh, or any value that selects a method, inside a `rank == 0` branch of
    your own. A rank holding `nothing` where the others hold a mesh dispatches to a different
    method, and the run then blocks forever waiting for a message that is never sent. Passing the
    reader to `distribute` rules this out.

For a large mesh, also give `dir`. Rank 0 decomposes the mesh into that directory once, and every
rank then loads only its own part, so after the first run no rank ever holds the global mesh:

```julia
mesh = distribute(dir="parts", key=("path/to/mesh.unv", 0.001)) do
    UNV3D_mesh("path/to/mesh.unv", scale=0.001)
end
```

If `dir` already holds a decomposition for the same number of ranks and the same `key`, it is reused
and the reader is never called; one for a different number of ranks, another key, or an older file
format is replaced. The key is any hashable value naming the mesh; without it only the rank count is
compared, so another mesh's parts left in `dir` would be reused. Parts cut from different meshes
mixed in one directory are refused at load.
[`partition_mesh`](@ref) writes the same layout from a standalone process if you want to decompose
ahead of time, and `distribute(dir; comm)` loads it. Each part is a binary `rank_<r>.xdm` file that
survives XCALibre and Julia upgrades; [`mesh_info`](@ref) reads its header (kind, rank count, integer
and float types, cell counts), and loading a part written for another rank count or format errors
with the call that fixes it.

### Reading a case decomposed by OpenFOAM

If the mesh is an OpenFOAM case already split with `decomposePar` (any method; OpenFOAM 12 and v2512
have been checked), each rank reads its own `processor<rank>` directory and the ghost cells are built
by one exchange between neighbours, so no process ever reads the whole mesh, not even once. Launch
with as many ranks as there are processor directories:

```julia
mesh = distribute(FOAMCase("path/to/case", scale=0.001))
```

`decomposePar` writes `cellProcAddressing`, so [`gather`](@ref) returns fields in the undecomposed
cell order, and results written with `output=OpenFOAM()` from the case directory reconstruct with
`reconstructPar`. A distributed run with `output=OpenFOAM()` also writes this layout, so its case can
be read back the same way. The example below writes a one-rank case and reads it:

```jldoctest distributed; filter = r".*"s => s"", output = false
box_file = joinpath(grids_dir, "3d_box_1000x1000x1000mm_5.unv")
case_dir = mktempdir()
cd(case_dir) do
    initialise_writer(OpenFOAM(), distribute(() -> UNV3D_mesh(box_file, scale=0.001)))
end
box_mesh = distribute(FOAMCase(case_dir))

# output

```

!!! note
    Writing results into a case read with `scale` other than 1 rewrites its processor meshes in
    metres; keep the original case if other OpenFOAM tools still need it.

### Rebalancing in parallel

A decomposition made by a geometric method such as `decomposePar -method simple` is valid but
poorly balanced. [`repartition`](@ref) partitions the distributed cell graph in parallel and moves
cells between ranks, again without any rank holding the whole mesh:

```julia
mesh = repartition(distribute(FOAMCase("path/to/case", scale=0.001)))
```

It calls PETSc's partitioners, so it needs `using PETSc` with a PETSc built with PT-Scotch (the
default, `method=:ptscotch`) or ParMETIS (`method=:parmetis`). The conda-forge `petsc` package has
both (see [GPU runs without compiling PETSc](@ref) for using it); `PETSc_jll` has neither, and the
call then errors. On the 3D backward-facing step at eight ranks, a `simple` decomposition with 5369
processor faces and a 1.23 ratio between the largest and smallest part became 1970 faces and 1.02
with PT-Scotch, slightly better than serial Metis (2046 faces, 1.04). ParMETIS gave 2089 faces but a
1.10 ratio.

### Which route to use

- **The mesh fits in one process**: `distribute() do ... end`. Rank 0 reads and partitions the mesh
  on every run; it needs about 0.9 KB per cell above the runtime, and more while partitioning (2.4 GB
  for 1.3 million cells).
- **Repeated runs of a large mesh that one process can still hold once**: `distribute(dir=...)`, or
  `partition_mesh` on a machine with more memory. The mesh is read once and later runs load only
  their parts.
- **A mesh no single node can hold, or one already decomposed in OpenFOAM**: `FOAMCase`, followed by
  `repartition` when the decomposition is geometric. The number of ranks must equal the number of
  processor directories.

## Setting up and running a case

Pass the distributed mesh as the model `domain` and assign boundary conditions against it, exactly
as in serial. The rest of the script is unchanged:

```jldoctest distributed; filter = r".*"s => s"", output = false
velocity = [0.5, 0.0, 0.0]
nu = 1e-3

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh
    )

BCs = assign(
    region = mesh,
    (
        U = [
            Dirichlet(:inlet, velocity),
            Extrapolated(:outlet),
            Wall(:wall, [0.0, 0.0, 0.0]),
            Symmetry(:top)
        ],
        p = [
            Extrapolated(:inlet),
            Dirichlet(:outlet, 0.0),
            Extrapolated(:wall),
            Symmetry(:top)
        ]
    )
)

solvers = (
    U = SolverSetup(
        solver = Bicgstab(),
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax = 0.7,
        rtol = 1e-1
    ),
    p = SolverSetup(
        solver = Cg(),
        preconditioner = GAMG(),
        convergence = 1e-7,
        relax = 0.3,
        rtol = 1e-2
    )
)

schemes = (
    U = Schemes(divergence = Linear),
    p = Schemes()
)

runtime = Runtime(iterations = 5, write_interval = -1, time_step = 1)
hardware = Hardware(backend = CPU(), workgroup = 1024)

config = Configuration(
    solvers = solvers, schemes = schemes, runtime = runtime, hardware = hardware, boundaries = BCs)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

residuals = run!(model, config)

is_root() && println("final pressure residual ", residuals.p[end])

# output

```

Two helpers cover what a parallel script still needs. [`is_root`](@ref), used above, is true on
rank 0. It is also true in a serial run where MPI was never initialised, so the same guard works in
both. `bind_device!(backend)` binds the calling rank to a GPU by its position among the ranks on
its own node, so the ranks on each node take that node's devices in turn; when a node has more
ranks than GPUs they share one and a warning says so.

Output on a distributed mesh is written with `output=OpenFOAM()` in the decomposed layout; VTK has
no decomposed writer, so a run that asks for VTK output with a positive `write_interval` stops with
an error at its first write. Leave `write_interval=-1` when no output is needed.

## Output, checkpoints and restart

With `output=OpenFOAM()` every rank writes `processor<rank>/<time>/` in the working directory, in
OpenFOAM's binary format: the mesh once under `constant/polyMesh`, and at each write the fields,
the face flux `phi` and a `uniform/time` file with the iteration and time step. Open the case in
ParaView through the `XCALibre.foam` file that rank 0 creates, or combine it with `reconstructPar`
when the decomposition came from `decomposePar`.

Every write is also a checkpoint. `write_interval` counts iterations (steady) or time steps
(transient), and a write happens whenever the iteration number is a multiple of it. To continue a
run, launch the same script on the same number of ranks, in the same directory, with `restart` set
to a written iteration (steady), time (transient) or time-directory name, and `iterations` set to
where the run should stop:

```julia
residuals = run!(model, config; output=OpenFOAM(), restart=500)   # resumes after iteration 500
```

The resumed run reads the velocity, pressure, face flux and turbulence fields (and the time and
time step of a transient run) that were written, and continues from the next iteration exactly as
the uninterrupted run would have: with Jacobi-preconditioned solves the residuals and fields match
bit for bit. The residual vectors it returns keep their initial fill for the iterations that were
not run. Restart is available for incompressible SIMPLE and PISO runs with `Laminar` and `KOmegaSST`
on a distributed mesh; a serial run started with `restart` stops with an error.

## Launching

Install MPI.jl's launcher once. It resolves the same MPI binary that the package itself uses:

```bash
julia --project=<env> -e 'using MPI; MPI.install_mpiexecjl()'
```

After that, a run is one command:

```bash
mpiexecjl -n 4 julia --project=<env> your_case.jl
```

`mpiexecjl` lives in `~/.julia/bin`, so add that directory to your `PATH`. You can set threads per
rank with Julia's usual `-t` flag, but one thread per rank is the right default. The CPU kernel
backend is already serial, so extra threads add overhead rather than saving time. Call
`activate_multithread(backend)` in the script. Despite its name, it pins BLAS to a single thread,
which stops each rank from taking every core. On a machine with hyperthreading, bind one rank per
physical core, for example `mpiexecjl -n 4 --bind-to core --map-by core julia ...`.

## PETSc solvers in a single process

The same script runs without `mpiexecjl`. Started with plain `julia your_case.jl`, `distribute`
initialises MPI with a single rank and every linear solve goes through PETSc. This is how you use
PETSc's solvers and preconditioners, such as `GAMG()`, in an ordinary serial run, as an alternative
to the Krylov.jl solvers used on a serial mesh. Your script does not need `using MPI`, since
XCALibre depends on it already.

On a GPU backend, the fields stay on the device and the linear solves need a GPU-enabled PETSc.

## Configuring the linear solvers

On a distributed mesh the solves go through PETSc. The `SolverSetup` fields keep their serial
meaning:

- `solver`: `Cg()` → `cg`, `Cgs()` → `cgs`, `Bicgstab()` → `bcgs`, `Gmres()` → `gmres` (PETSc
  `KSP` type).
- `preconditioner`: `Jacobi()` → `jacobi`, `GAMG()` → `gamg`, `BoomerAMG()` → `hypre` (PETSc
  `PC` type). `DILU()`, `ILU0GPU()` and `IC0GPU()` have no PETSc equivalent and map to their
  closest relative, block Jacobi with an ILU(0) (`DILU`, `ILU0GPU`) or ICC(0) (`IC0GPU`)
  factorisation of each rank's block. Each warns once that it is a substitute.
- `atol`, `rtol`, `itmax`: the stopping tolerances and iteration limit of each linear solve, as in
  serial.
- `convergence`: the residual target that stops the outer iteration, as in serial. It does not
  affect the linear solver.

When each solver is built, rank 0 prints the configuration PETSc received:

```
[ Info: PETSc solve [p]: KSP=cg PC=gamg atol=1.0e-6 rtol=0.01 itmax=1000 reuse_interpolation=true
```

The `petsc_options` keyword of `run!` passes a raw PETSc options string, which reaches every
solve. Use it for anything the list above does not cover. It overrides the mapped types, and it
also names a type for a solver or preconditioner that has no mapping:

```julia
run!(model, config; petsc_options="-pc_type sor -ksp_monitor")
```

To configure one equation differently, pass a named tuple keyed by the equation's label instead:
`all` applies to every solve, and `U`, `p`, `k`, `omega`, `T` or `y` (wall distance) apply to that
equation only, after `all`:

```julia
run!(model, config; petsc_options = (all = "-log_view", U = "-pc_type asm -sub_pc_type ilu"))
```

PETSc also reads these options at start-up, so options that must be set before PETSc initialises,
such as `-log_view` or `-use_gpu_aware_mpi 0`, go there too and need no environment variable. Put
them in the plain string or in `all`: PETSc initialises once per process, so start-up options take
effect only with the first solver built, and an entry for a later equation (for example `p` when
`U` is built first) cannot add them.

## Choosing a preconditioner

| Preconditioner | Use it for | Avoid it when |
|---|---|---|
| `Jacobi()` | momentum and turbulence; pressure on small partitions; comparing runs across rank counts, since its results do not depend on the partition | the pressure mesh is large: its iteration count grows with mesh size |
| `GAMG()` | pressure on large meshes; the default AMG, since it needs no special build | the operator is not symmetric positive definite (momentum, turbulence), or each rank holds a small partition, where setup is not repaid |
| `BoomerAMG()` | pressure where GAMG converges poorly, or when the strongest reduction per solve matters | Float32 or builds without hypre; small partitions; you need results that match across rank counts |
| `DILU()`, `ILU0GPU()`, `IC0GPU()` | running serial scripts unchanged; each becomes a per-rank incomplete factorisation and warns once | you expect the serial method itself: the substitute weakens as ranks are added, because each block ignores its neighbours |

For momentum and turbulence, `Bicgstab()` with `Jacobi()` is the usual choice. For pressure, start
with `Cg()` and `Jacobi()` on small cases and `Cg()` and `GAMG()` on large ones.

## Algebraic multigrid for pressure (GAMG and BoomerAMG)

The pressure Poisson solve dominates incompressible runs. Its condition number worsens as the mesh
grows, so Jacobi-preconditioned CG needs more iterations on larger meshes. Algebraic multigrid
(AMG) builds a hierarchy of coarser problems and converges the pressure in a number of Krylov
iterations that is roughly independent of mesh size. Two AMG preconditioners are available. Both
are for symmetric positive-definite systems only (they have no transpose apply), and both run only
through PETSc:

- `GAMG()`: PETSc's native aggregation AMG (`-pc_type gamg`). It needs no extra build flags.
- `BoomerAMG()`: HYPRE BoomerAMG (`-pc_type hypre`). It needs a PETSc built with hypre; the stock
  Float64 libraries include it.

```julia
p = SolverSetup(solver = Cg(), preconditioner = GAMG(), convergence = 1e-7, relax = 0.3, rtol = 0.01)
```

**Prefer `GAMG()` when you want AMG.** It needs no special build, and on the cases measured so far
it was faster than `BoomerAMG()` at every rank count and scaled more evenly (see
[Benchmarks](@ref)). `BoomerAMG()` remains available for operators where classical coarsening
suits better.

!!! note "AMG results depend on the rank count"
    An AMG hierarchy is built from each rank's local partition. It therefore changes with the rank
    count, and runs at different rank counts converge to the same solution along slightly
    different paths. `Jacobi()` does not depend on the partition and gives identical residuals at
    every rank count, so use it when you need to compare runs at different rank counts exactly.

### When AMG helps

AMG adds a cost to every solve that Jacobi does not have: building or refreshing the hierarchy,
then applying a V-cycle. On small partitions this can make AMG *slower* than Jacobi. Jacobi's
iteration count grows with mesh size, roughly as the cube root of the cell count in 3D, while AMG's
stays nearly constant, so AMG overtakes Jacobi above some problem size. That crossover depends on
the mesh, the rank count and the memory system, so measure it on your own case. Adding ranks
shrinks each partition and erodes AMG's advantage, because its coarse levels become
communication-bound.

AMG also stores its hierarchy of coarse operators, so each rank needs noticeably more memory than
with Jacobi. On a memory-limited machine this, not time, can set how many ranks a large mesh can
use. In return, AMG reduces the pressure residual much further in each outer iteration, which can
cut the number of outer iterations needed to reach a steady state. As a rule of thumb, use Jacobi
for small and medium cases and AMG for large ones, especially when the pressure solve dominates.

### Memory per rank

Every rank pays a fixed cost of roughly 0.8 GB for the Julia runtime, the loaded packages and the
compiled code, plus a cost per cell it holds. For a laminar Float64 case with Jacobi the per-cell
cost is about 1.8 KB, so a rank holding 660,000 cells peaks near 1.8 GB. Turbulence models, AMG and
a finer mesh near the wall raise it.

Not all of the fixed cost is paid once per rank. About 250 MB is the Julia system image, package
images and shared libraries, which the operating system holds once per node however many ranks map
them. The rest, about 0.55 GB, is private to each rank. Of that, roughly 200 MB is memory that the
garbage collector keeps after compiling the solver on the first iteration. A node running `n` ranks
therefore needs about `0.25 GB + n × (0.55 GB + per-cell cost)`. Tools that sum each process's
resident memory count the shared part `n` times. The proportional set size (`Pss` in
`/proc/<pid>/smaps_rollup`) does not.

Each rank's garbage collector sizes its heap without knowing
about the other ranks on the node, so set a heap size hint per rank of roughly the node's memory
divided by the ranks on it, leaving room for PETSc, which allocates outside the Julia heap. Precompile
in a plain session first, since a rank that must precompile under a different flag fails to load:

```bash
mpiexecjl -n 8 julia --heap-size-hint=1500M --project my_case.jl
```

The hint must be given at launch, as the flag or as the `JULIA_HEAP_SIZE_HINT` environment
variable. Setting it later from the script cannot give back the pages used while loading. A tight
hint makes the collector run more often. On small meshes it lowers the first-run peak by up to 30
percent, but it can slow each iteration by 10 to 20 percent, so leave it loose unless memory,
not time, is the limit.

### Precompiling a production case

Most of the first run's private memory and compile time goes on compiling the solver for your
case's exact types. When the same case is run many times, you can compile it once, ahead of time,
into a small local package. On a 10 mm BFS case at
four ranks this lowered private memory per rank from 559 to 359 MB and cut first-run compilation
from 11.5 s to under 0.1 s. Compiled kernels do not depend on the mesh size or the rank count, so
the package stays valid when the case is refined or run on more ranks: traced at two ranks on the
10 mm mesh, it still cut first-run compilation from 10.2 to 0.4 s at four ranks, and gave the same
saving on the 5 mm mesh. Boundary values, iteration counts and relaxation factors can change too.
Changing the physics models, the boundary conditions (their types, or the patches they apply to),
the schemes, the solvers or the integer and float types does not keep it valid, so trace the case
again after any of those.

1. Trace one short run (two iterations are enough), on a coarse mesh and a few ranks if you like;
   use at least two ranks so the exchange paths are traced. The wrapper names one trace file per
   rank (MPICH sets `PMI_RANK`, Open MPI `OMPI_COMM_WORLD_RANK`):

   ```bash
   cat > trace.sh <<'EOF'
   #!/bin/sh
   exec julia --project=<env> --trace-compile="trace_${PMI_RANK:-$OMPI_COMM_WORLD_RANK}.jl" "$@"
   EOF
   chmod +x trace.sh
   mpiexecjl -n 4 ./trace.sh my_case.jl
   ```

2. Create a package from the traces. Its dependencies are copied from the case's environment, so
   that it compiles against exactly the package versions the trace ran with. Statements that name
   your script's own definitions (`Main.`) are dropped:

   ```bash
   julia --project=<env> -e 'using Pkg, TOML
       Pkg.generate("CasePrecompile")
       deps = TOML.parsefile(Base.active_project())["deps"]
       p = TOML.parsefile("CasePrecompile/Project.toml")
       p["deps"] = Dict(k => deps[k] for k ∈ ("XCALibre", "PETSc", "MPI"))
       open(io -> TOML.print(io, p), "CasePrecompile/Project.toml", "w")'
   cat trace_*.jl | grep -v 'Main\.' | sort -u > CasePrecompile/src/statements.jl
   ```

   and replace `CasePrecompile/src/CasePrecompile.jl` with:

   ```julia
   module CasePrecompile
   using XCALibre, PETSc, MPI
   for (id, m) ∈ Base.loaded_modules # make every loaded module nameable in the statements
       isdefined(@__MODULE__, Symbol(id.name)) || Core.eval(@__MODULE__, :(const $(Symbol(id.name)) = $m))
   end
   const XCALibrePETScExt = Base.get_extension(XCALibre, :XCALibrePETScExt)
   include_dependency(joinpath(@__DIR__, "statements.jl"))
   for line ∈ eachline(joinpath(@__DIR__, "statements.jl"))
       try Core.eval(@__MODULE__, Meta.parse(line)) catch end
   end
   end
   ```

3. Add it to the case's environment without changing any other package's version, precompile it in
   a plain session, and load it at the top of the case script, before the mesh is distributed. If
   the environment is updated later, trace again, since the traced types may no longer match:

   ```bash
   julia --project=<env> -e 'using Pkg; Pkg.develop(path="CasePrecompile"; preserve=Pkg.PRESERVE_ALL); Pkg.precompile()'
   ```

   ```julia
   using XCALibre, PETSc, MPI
   using CasePrecompile
   ```

### Rebuilding vs freezing the hierarchy

In SIMPLE, the pressure matrix keeps a fixed sparsity pattern while its coefficients change
slightly from one outer iteration to the next. Rebuilding the whole AMG hierarchy for every solve
is expensive, so both preconditioners avoid it, in different ways:

- `GAMG()` sets `reuse_interpolation = true` by default. It builds the aggregation and
  interpolation operators once and then, for each solve, recomputes only the coarse operators and
  smoothers, which is cheap. The hierarchy stays numerically current at a fraction of the cost of
  a full setup. This is valid only because the sparsity pattern never changes. `GAMG(freeze = N)`
  goes further and holds the whole preconditioner fixed for `N` solves, skipping even that update
  (default `25`; `freeze = 1` updates for every solve).
- `BoomerAMG(freeze = N)` holds the whole preconditioner fixed for `N` solves and rebuilds it from
  the current matrix on the `N`th (default `10`; `freeze = 1` rebuilds for every solve). HYPRE
  cannot partially reuse a hierarchy, so this all-or-nothing freeze is the only option.

The Krylov solver always uses the current matrix, so a frozen preconditioner changes how fast each
solve converges but not the solution it converges to. A longer freeze saves setup time and can cost
residual reduction per outer iteration. The defaults balance the two.

### Tuning keywords

Each keyword `k = v` is forwarded to PETSc and overrides a default.

**GAMG** → `-pc_gamg_<k> v`. Common options:

- `threshold`: aggregation strength threshold (e.g. `0.01`–`0.05`).
- `agg_nsmooths`: prolongator smoothing steps; `0` gives unsmoothed aggregation, which is cheaper
  and often good for Poisson problems.
- `reuse_interpolation`: reuse aggregation and interpolation across solves (default `true` here).
- `coarse_eq_limit`: the size at which the coarsest level is solved directly.

```julia
GAMG(threshold = 0.02, agg_nsmooths = 0)
```

**BoomerAMG** → `-pc_hypre_boomeramg_<k> v`. The defaults are tuned for 3D (`strong_threshold =
0.7`, `coarsen_type = "HMIS"`, `interp_type = "ext+i"`, `P_max = 4`, `agg_nl = 1`,
`agg_num_paths = 2`). PETSc's own BoomerAMG defaults (Falgout coarsening, classical interpolation,
unlimited interpolation stencil) build an overly complex, memory-heavy hierarchy in 3D. Common
options:

- `strong_threshold`: strength-of-connection threshold; 0.5–0.7 for 3D.
- `coarsen_type`: coarsening algorithm, such as `"HMIS"`, `"PMIS"` or `"Falgout"`.
- `interp_type`: interpolation, such as `"ext+i"` or `"classical"`.
- `P_max`: maximum interpolation entries per row, which bounds operator complexity.
- `agg_nl`: number of aggressive-coarsening levels, which lowers operator complexity and memory.
- `relax_type_all`: smoother, such as `"SOR/Jacobi"`, `"Chebyshev"` or `"l1scaled-Jacobi"`.
- `grid_sweeps_all`: smoother sweeps per level.

```julia
BoomerAMG(strong_threshold = 0.6, coarsen_type = "PMIS", relax_type_all = "Chebyshev", freeze = 20)
```

See PETSc's `-pc_gamg_*` and `-pc_hypre_boomeramg_*` option lists for the full set. Anything not
exposed as a keyword can still be passed through `petsc_options`.

!!! note
    `BoomerAMG` needs a PETSc with hypre. The stock Float64 libraries have it, but Float32 and many
    custom builds do not, and there the solver raises an error when it is constructed. `GAMG`
    needs no special build and is the safer default in shared scripts.

### On the GPU

With a CUDA-enabled PETSc, `Jacobi()` and `GAMG()` run on the device and give the same residuals
as on the CPU; GAMG builds part of its hierarchy on the host, which its `freeze` count amortises.
`BoomerAMG()` runs on the device only if PETSc's hypre was itself built with CUDA. There PETSc
switches it to the GPU-capable variants (PMIS coarsening, `ext+i` interpolation, l1-Jacobi
smoothing), so its residuals differ from a CPU run with the same keywords. With a PETSc whose
hypre runs on the host only, such as the conda-forge build, `BoomerAMG()` on GPU fields would
crash, so XCALibre asks hypre for its execution policy first and stops with an error naming
`GAMG()` and `Jacobi()` instead; `BoomerAMG(device=true)` skips that check for a build the query
misreads. `GAMG()` is the recommended AMG on the GPU.

### Other PETSc preconditioners

Any PETSc preconditioner can be named through `petsc_options`, for every solve or, with a named
tuple, for one equation. Three are worth knowing:

- Additive Schwarz with ILU subdomains, the robust general-purpose choice for non-symmetric
  systems such as momentum and turbulence: `-pc_type asm -pc_asm_overlap 1 -sub_pc_type ilu`.
- Local SOR/SSOR sweeps, cheap and often adequate for momentum: `-pc_type sor`.
- HPDDM, a multilevel domain-decomposition method for hard, strongly anisotropic problems at large
  rank counts. It needs a PETSc built with HPDDM: `-pc_type hpddm`.


## Terminal output

Informational `@info` messages are printed once, from rank 0, rather than once per rank. Warnings
and errors still come from every rank, so failures on a single rank remain visible. This happens
automatically when the mesh is distributed, and your script needs nothing extra.

## What is and is not supported

Distributed today:

- Steady and transient incompressible flow through the SIMPLE and PISO families.
- The Laplace (conduction) solver, and `potential_flow!` for initialising a velocity field.
- `Laminar`, `KOmega` and `KOmegaSST` turbulence, including wall distance.
- CPU and GPU backends, periodic patches, and writing results in OpenFOAM's decomposed layout so
  that the usual tools can reconstruct them.

Not distributed yet. These raise an error naming every missing piece, rather than silently giving a
wrong answer: without a distributed linear-solve seam each rank would solve only its own block.

- Compressible flow (`csimple!`, `cpiso!`), the density-based supersonic solver, multiple reference
  frames, the film model and the multiphase solver.
- The `KOmegaLKE` transition model and the LES models.
- Float32 with `BoomerAMG`, because the stock PETSc libraries include hypre at Float64 only.
- GPU runs without a CUDA-enabled PETSc build (see [Setting up MPI and PETSc](@ref)).
- AMD GPUs, because PETSc.jl cannot yet hand PETSc's HIP vectors back as device arrays.

The supported set is declared by `distributed_ready` methods in `src/Solvers/Solvers_0_functions.jl`,
which default to unsupported, so a newly added solver or model is refused until it has been wired
and tested.

## What to expect from parallel performance

- **Measure with a steady clock.** Many CPUs, laptops in particular, lower their clock frequency as
  more cores become busy. This alone can look exactly like poor scaling, so pin the frequency or
  hold package power constant before comparing timings across rank counts.
- **Prefer fewer, larger subdomains.** Every Krylov iteration ends in a global reduction. With many
  small subdomains, the ranks spend a growing share of each iteration waiting at that reduction
  rather than computing.
- **Stop at one rank per physical core.** Beyond that, ranks compete for the same execution
  resources and gain nothing.
