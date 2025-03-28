initialise_writer(format::OpenFOAM, mesh::Mesh3) = FOAMWriter(nothing, nothing)
initialise_writer(format::OpenFOAM, mesh) = error("
The OpenFOAM format can only be used for 3D simulations. Use `output=VTK()` instead.
")

function write_results(iteration, mesh, meshData::FOAMWriter, args...)
    timedir = @sprintf "%i" iteration
    mkpath(timedir)
end