# phase 3: rank-aware logging (#3)
## goal
Each @info shows once (from rank 0), not once per rank. Zero per-solver guards.
## design
- In distribute() (both mesh and dir methods, after MPI.Init + rank known): if this rank != 0,
  install a global logger that suppresses Info/Debug but passes Warn/Error. Use stdlib Logging:
  `global_logger(ConsoleLogger(stderr, Logging.Warn))` on non-root. Keep root default.
  ponytail: stdlib ConsoleLogger with min-level Warn — no custom logger type.
- Do it once; idempotent (only when nranks>1 and rank!=0).
- Crash/error visibility preserved (Warn/Error still print from every rank).
## steps
- [ ] set non-root global logger in distribute() (guard nranks>1)
- [ ] remove the temporary rank==0 guard added in phase 1 echo (logger now handles it)
- [ ] gate: n=4 run, grep count of a known @info line == 1
## gate
n=4 psimple gate: "Starting SIMPLE" (or resolved-config line) appears exactly once in the log;
a deliberately-triggered @warn still appears from a non-root rank.
## risks/assumptions
- MPI programs share stdout/stderr; min-level filter is the lever, ordering not guaranteed.
- Must not swallow @error (needed for debugging rank-local crashes).
