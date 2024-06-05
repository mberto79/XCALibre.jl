Chris' Notes

Compressibility is now a part of a single solver including both transonic and subsonic flows. - Solvers_1_SIMPLE_RHO_K_transonic.jl - ignore the other additions

Three test cases are setup to demonstrate the needs:
    - CASE_UNV_flatplate_fixedQ.jl - subsonic heated plate
    - CASE_cylinder_fixedQ.jl - subsonic heated cylinder
    - CASE_cylinder_fixedQ_transonic.jl - supersonic cylinder

Add documentation and FVM background so the whole thing makes sense