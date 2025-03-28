initialise_writer(format::OpenFOAM, mesh::Mesh3) = FOAMWriter(nothing, nothing)

function write_results(iteration, mesh, meshData::FOAMWriter, args...)
    name = @sprintf "time_%.6d" iteration
    filename = name*".vtk"
    println(filename)
end