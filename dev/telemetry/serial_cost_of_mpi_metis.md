# What MPI and Metis cost a user who never runs distributed

Measured 2026-09-17, Julia 1.13, `dev/petscenv_stock`, fresh process per figure.

| quantity | value |
|---|---:|
| `using XCALibre`, everything included | 0.809 s |
| `import MPI` alone in a fresh process | 0.211 s |
| `using XCALibre` with MPI already loaded | 0.639 s |
| MPI's share of the package load | 0.17 s |
| `MPICH_jll` artifact tree | 24.0 MB |
| `METIS_jll` artifact tree | 2.9 MB |

So a serial user pays roughly a fifth of a second and 27 MB, about a fifth of the package's load
time, for dependencies they will not call.

## What removing it would cost

`HaloExchange` names `MPI.Comm` and `Vector{MPI.Request}` in its fields, so the type would have
to be parameterised, and every file in `src/Distribute/` calls MPI, so the bodies would move to
an extension while the types, exported names and the seams `Solvers` dispatches on stayed behind.
`XCALibrePETScExt` imports the distributed solver type and its assemble and solve functions by
name, and the package re-exports the submodule, so both would need care. Six steps against a
fifth of a second.
