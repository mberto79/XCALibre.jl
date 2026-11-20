* changed: All transient solvers set their time step through one shared time-step control; each solver supplies only its Courant criteria and time-step limits.
* changed: With `AdaptiveTimeStepping` the `SupersonicFlow` solver now grows its time step by at most `maxGrow` per iteration; it still never exceeds the CFL limit.
* changed: `LocalTimeStepping` with a solver other than the `SupersonicFlow` solver now throws an `ArgumentError` at start-up.
* fixed: The film solver no longer fails when a post-processing start time is given.
* added: `Runtime(adaptive=...)` accepts any subtype of `XCALibre.Solve.AbstractTimeStepping` implementing `update_dt!`, so solvers can be given new time-stepping modes.
* changed: Writing `runtime.dt` during a run no longer changes the time step; use a time-stepping mode instead.
