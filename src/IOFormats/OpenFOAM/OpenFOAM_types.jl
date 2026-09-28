export FOAMWriter
export attach_state!

struct FOAMWriter{H,F}
    header::H
    footer::F
end

# solvers hand the loop state that is not a model field (face flux, time step) to the writer; only
# writers that output it (decomposed OpenFOAM) keep it
attach_state!(writer, flux, dt) = nothing
