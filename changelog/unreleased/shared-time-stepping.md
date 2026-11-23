* changed: All transient solvers set their time step through one shared time-step control; each solver supplies only its Courant criteria and time-step limits.
* changed: With `AdaptiveTimeStepping` the `SupersonicFlow` solver now grows its time step by at most `maxGrow` per iteration; it still never exceeds the CFL limit.
* changed: `LocalTimeStepping` with a solver other than the `SupersonicFlow` solver now throws an `ArgumentError` at start-up.
* fixed: The film solver no longer fails when a post-processing start time is given.
* breaking: `Runtime` no longer takes `adaptive`; the time-stepping mode is passed as `time_step`: an integer (steady), a number (fixed time step), `AdaptiveTimeStepping(dt=..., maxCo=...)` or `LocalTimeStepping(...)`. `AdaptiveTimeStepping` now requires the initial time step `dt`.
* changed: Steady solvers reject a time-stepping mode with an `ArgumentError` and warn when `time_step` is not an integer.
* added: `Runtime(time_step=...)` accepts any subtype of `XCALibre.Solve.AbstractTimeStepping` implementing `update_dt!` (and `initial_time_step` unless it has a `dt` field), so solvers can be given new time-stepping modes.
* changed: Writing `runtime.dt` during a run no longer changes the time step; use a time-stepping mode instead.
