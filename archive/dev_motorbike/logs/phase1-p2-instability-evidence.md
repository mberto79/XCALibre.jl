# P2 instability evidence

- XCALibre iteration 50: max `|U| = 1.18906e5`, max `p = 1.42719e9` at one-based cell 210839.
- XCALibre iteration 100: max `|U| = 1.043e8`; iteration 200: max `|U| = 1.63e13`.
- OpenFOAM time 500 at the same mesh cell: `|U| ~= 6.38`, `p ~= 99.6`.
- Cell 210839 owns three motorbike wall faces: two on `motorBike_frt-fairing:001%1` and one on `motorBike_frt-fairing:001-shadow%74`.
- The neighboring cell across internal face 630057 carries the `k`/`nut` peak.
- No slip face is incident to the failing cell. The initiating topology instead exercises repeated wall-function writes and the wall velocity Laplacian.
- Source audit: `Wall Laplacian{Linear} VectorField` retains owner-cell normal velocity for a stationary wall; `_set_production!` and `_constrain!` launch per face and overwrite cell values/rows for multiply incident wall faces.
