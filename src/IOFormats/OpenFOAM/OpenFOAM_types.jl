export FOAMWriter
export attach_flux!

struct FOAMWriter{H,F}
    header::H
    footer::F
end

# solvers hand their face flux to the writer; only writers that output it (decomposed OpenFOAM) keep it
attach_flux!(writer, flux) = nothing
