# phase 4: curated BoomerAMG tuning + docs (#4 D3 + #1 docs)
## goal
Discoverable HYPRE tuning without raw strings; a distributed docs page.
## design
- BoomerAMG curated (D3-B): change struct BoomerAMG to hold opts NamedTuple with keyword ctor:
    struct BoomerAMG{NT} <: PreconditionerType; opts::NT end
    BoomerAMG(; kwargs...) = BoomerAMG(values(kwargs))
  Keep _pc_type(::BoomerAMG)="hypre" and the serial-error Preconditioner{BoomerAMG} ctor
  (adjust to BoomerAMG{NT} match). In XCALibrePETScExt, expand setup.preconditioner.opts into
  PETSc options: map curated kwargs → -pc_hypre_boomeramg_* (strong_threshold, coarsen_type,
  relax_type, etc). Only map a small documented allowlist; anything else still via petsc_options.
- Docs: docs/src distributed page — launch (mpiexec incantation + reduced script), solver
  config (what maps to PETSc, convergence fallback), BoomerAMG-for-pressure recommendation +
  the curated knobs, PETSc build requirement (--download-hypre).
## steps
- [ ] BoomerAMG{NT} + keyword ctor; fix serial error ctor + _pc_type
- [ ] petsc opts expansion for the curated allowlist
- [ ] docs page + link in docs nav
- [ ] gate
## gate
psimple MPI gate with preconditioner=BoomerAMG(strong_threshold=0.7) runs n=2 and the option
reaches PETSc (grep KSPView/log or assert no error). docs/make.jl builds clean.
## risks/assumptions
- Adapt.@adapt_structure not needed (BoomerAMG never goes to device as a field).
- kwargs allowlist small; document that full control stays in petsc_options.
