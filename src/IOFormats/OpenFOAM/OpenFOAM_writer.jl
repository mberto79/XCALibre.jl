
initialise_writer(format::OpenFOAM, mesh::Mesh3) = begin
    # create dummy file to load results in ParaView
    touch("XCALibre.foam")
    default_dir = "constant/polyMesh"

    if !isdir(default_dir)
        @info "Writing mesh to constant/polyMesh..."
        # Create constant directory and mesh files
        polyMeshDir = mkpath(default_dir)
        pointsFile = joinpath(polyMeshDir, "points")
        facesFile = joinpath(polyMeshDir, "faces")
        ownerFile = joinpath(polyMeshDir, "owner")
        neighbourFile = joinpath(polyMeshDir, "neighbour")
        boundaryFile = joinpath(polyMeshDir, "boundary")
        backend = _get_backend(mesh)

        # Copy mesh data and get basic SteadyState
        nodes = get_data(mesh.nodes, backend) # get cpu copy
        cells = get_data(mesh.cells, backend) # get cpu copy
        faces = get_data(mesh.faces, backend) # get cpu copy
        face_nodes = get_data(mesh.face_nodes, backend) # get cpu copy
        boundaries = get_data(mesh.boundaries, backend) # get cpu copy
        npoints = length(nodes)
        ncells = length(cells)
        nfaces = length(faces)
        bfaces = length(mesh.boundary_cellsID)
        ifaces = nfaces - bfaces

        # write points 
        
        open(pointsFile, "w") do io
            println(io, 
            """
            FoamFile
            {
                version     2.0;
                format      ascii;
                class       vectorField;
                location    "constant/polyMesh";
                object      points;
            }
            """)
            println(io, npoints)
            println(io, "(")
            for nodei ∈ eachindex(nodes)
                coords = nodes[nodei].coords
                println(io, "($(coords[1]) $(coords[2]) $(coords[3]))")
            end
            println(io, ")")
        end

        # write faces 
        open(facesFile, "w") do io
            println(io, 
            """
            FoamFile
            {
                version     2.0;
                format      ascii;
                class       faceList;
                location    "constant/polyMesh";
                object      faces;
            }
            """)
            println(io, length(faces))
            println(io, "(")
            # loop over internal faces first
            for fID ∈ (bfaces + 1):nfaces
                nrange = faces[fID].nodes_range
                nodesID = @view face_nodes[nrange]
                write(io, "$(length(nrange))(")
                for nID ∈ nodesID
                    foam_nID = nID - 1 # FOAM is zero-indexed
                    write(io, "$foam_nID ")
                end
                write(io, ")\n")
            end

            # loop over boundary faces at the end
            for fID ∈ 1:bfaces
                nrange = faces[fID].nodes_range
                nodesID = @view face_nodes[nrange]
                write(io, "$(length(nrange))(")
                for nID ∈ nodesID
                    foam_nID = nID - 1 # FOAM is zero-indexed
                    write(io, "$foam_nID ")
                end
                write(io, ")\n")
            end
            println(io, ")")
        end

        # write owners 
        open(ownerFile, "w") do io
            println(io, 
            """
            FoamFile
            {
                version     2.0;
                format      ascii;
                class       labelList;
                note        "nPoints: $npoints nCells: $ncells nFaces: $nfaces nInternalFaces: $ifaces";
                location    "constant/polyMesh";
                object      owner;
            }
            """)
            println(io, length(faces))
            println(io, "(")
            # loop over internal faces first
            for fID ∈ (bfaces + 1):nfaces
                owner = faces[fID].ownerCells[1] - 1 # OF uses zero index
                write(io, "$owner\n")
            end

            # loop over boundary faces at the end
            for fID ∈ 1:bfaces
                owner = faces[fID].ownerCells[1] - 1 # OF uses zero index
                write(io, "$owner\n")
            end
            println(io, ")")
        end

        # write neighbours 
        open(neighbourFile, "w") do io
            println(io, 
            """
            FoamFile
            {
                version     2.0;
                format      ascii;
                class       labelList;
                note        "nPoints: $npoints nCells: $ncells nFaces: $nfaces nInternalFaces: $ifaces";
                location    "constant/polyMesh";
                object      neighbour;
            }
            """)
            println(io, ifaces)
            println(io, "(")
            # loop over internal faces only
            for fID ∈ (bfaces + 1):nfaces
                neighbour = faces[fID].ownerCells[2] - 1 # OF uses zero index
                write(io, "$neighbour\n")
            end
            println(io, ")")
        end

        # write boundary 
        open(boundaryFile, "w") do io
            println(io, 
            """
            FoamFile
            {
                version     2.0;
                format      ascii;
                class       polyBoundaryMesh;
                location    "constant/polyMesh";
                object      boundary;
            }
            """)
            println(io, length(boundaries))
            println(io, "(")
            # loop over boundaries
            for boundary ∈ boundaries
                name = boundary.name
                IDs_range = boundary.IDs_range
                patchFaces = length(IDs_range)
                startFace = IDs_range[1] + ifaces - 1 # FOAM is zero-indexed
                write(io, """
                $name
                {
                    type            patch;
                    nFaces          $patchFaces;
                    startFace       $startFace;
                }
                """)
            end
            println(io, ")")
        end
    else
        @info "Mesh file already exsists in constant/polyMesh..."
    end

    # return dummy structure for dispatch
    FOAMWriter(nothing, nothing)
end

initialise_writer(format::OpenFOAM, mesh) = error("
The OpenFOAM format can only be used for 3D simulations. Use `output=VTK()` instead.
")

function write_results(iteration::TI, mesh, meshData::FOAMWriter, args...) where TI
    timedir = ""
    if TI <: Integer
        timedir = @sprintf "%i" iteration
    else
        timedir = @sprintf "%.8f" iteration
    end

    timedirpath = mkpath(timedir)

    backend = _get_backend(mesh)
    boundaries_cpu = get_data(mesh.boundaries, backend) # get cpu copy

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
                for BC ∈ field.BCs
                    println(io, "\t", boundaries_cpu[BC.ID].name)
                    println(io, _foam_boundary_entry(BC))
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
                for BC ∈ field.BCs
                    println(io, "\t", boundaries_cpu[BC.ID].name)
                    println(io, _foam_boundary_entry(BC))
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

_foam_boundary_entry(BC) = begin # catch all method
    """
    \t{
    \t\ttype zeroGradient;
    \t}
    """
end

_foam_boundary_entry(BC::Neumann) = begin
    """
    \t{
    \t\ttype zeroGradient;
    \t}
    """
end

_foam_boundary_entry(BC::Symmetry)  =  begin
    """
    \t{
    \t\ttype zeroGradient;
    \t}
    """
end

_foam_boundary_entry(BC::Dirichlet{ID,Value}) where {ID,Value<:Number} =  begin
    """
    \t{
    \t\ttype fixedValue;
    \t\tvalue uniform $(BC.value);
    \t}
    """
end

_foam_boundary_entry(BC::Dirichlet{ID,Value}) where {ID,Value<:SVector} =  begin
    value = BC.value
    """
    \t{
    \t\ttype fixedValue;
    \t\tvalue uniform ($(value[1]) $(value[2]) $(value[3]));
    \t}
    """
end

_foam_boundary_entry(BC::Wall{ID,Value}) where {ID,Value<:Number} =  begin
    """
    \t{
    \t\ttype zeroGradient;
    \t}
    """
end

_foam_boundary_entry(BC::Wall{ID,Value}) where {ID,Value<:SVector} =  begin
    value = BC.value; ux = value[1]; uy = value[2]; uz = value[3]
    """
    \t{
    \t\ttype fixedValue;
    \t\tvalue uniform ($ux $uy $uz);
    \t}
    """
end

