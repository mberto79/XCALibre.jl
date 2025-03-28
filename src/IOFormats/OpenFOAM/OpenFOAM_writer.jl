
initialise_writer(format::OpenFOAM, mesh::Mesh3) = FOAMWriter(nothing, nothing)

initialise_writer(format::OpenFOAM, mesh) = error("
The OpenFOAM format can only be used for 3D simulations. Use `output=VTK()` instead.
")

function write_results(iteration, mesh, meshData::FOAMWriter, args...)
    timedir = @sprintf "%i" iteration
    timedirpath = mkpath(timedir)

    backend = _get_backend(mesh)
    cells_cpu, faces_cpu, boundaries_cpu = copy_to_cpu(
        mesh.cells, mesh.faces, mesh.boundaries, backend)

    for arg ∈ args
        label = arg[1]
        field = arg[2]
        filename = joinpath(timedirpath, label)
        field_type = typeof(field)
        if field_type <: ScalarField
            open(filename, "w") do io
                write(io,"""
                FoamFile
                {
                    version     2.0;
                    format      ascii;
                    class       volScalarField;
                    location    "$iteration";
                    object      $label;
                }
                
                """)
                write(io, "internalField   nonuniform List<scalar>\n")
                println(io, length(mesh.cells))
                println(io, "(")
                values_cpu = copy_scalarfield_to_cpu(field.values, backend)
                for value ∈ values_cpu
                    println(io, value)
                end
                println(io, ");")

                println(io, "boundaryField")
                println(io, "{")
                for boundary ∈ boundaries_cpu
                    println(io, "\t", boundary.name)
                    println(io, "\t{")
                    println(io, "\t\ttype zeroGradient;")
                    println(io, "\t}")
                end
                println(io, "}")
            end
        elseif field_type <: VectorField
            open(filename, "w") do io
                write(io,"""
                FoamFile
                {
                    version     2.0;
                    format      ascii;
                    class       volVectorField;
                    location    "$iteration";
                    object      $label;
                }
                
                """)
                write(io, "internalField   nonuniform List<vector>\n")
                println(io, length(mesh.cells))
                println(io, "(")                
                x_cpu, y_cpu, z_cpu = copy_to_cpu(field.x.values, field.y.values, field.z.values, backend)
                for i ∈ eachindex(x_cpu)
                    println(io, "(",x_cpu[i]," ", y_cpu[i] ," ", z_cpu[i], ")")
                end
                println(io, ");")

                println(io, "boundaryField")
                println(io, "{")
                for boundary ∈ boundaries_cpu
                    println(io, "\t", boundary.name)
                    println(io, "\t{")
                    println(io, "\t\ttype zeroGradient;")
                    println(io, "\t}")
                end
                println(io, "}")
            end
        else
            throw("""
            Input data should be a ScalarField or VectorField e.g. ("U", U)
            """)
        end
    end
end