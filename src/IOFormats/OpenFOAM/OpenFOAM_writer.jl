
function _foam_declared_count(file_path)
    header = open(io -> String(read(io, 16384)), file_path) # counts sit just after the header
    header = replace(header, r"(?s)/\*.*?\*/" => " ")
    header = replace(header, r"//[^\r\n]*" => " ")
    count = nothing
    depth = 0
    for token ∈ eachmatch(r"\"(?:\\.|[^\"])*\"|[{}();]|[^\s{}();]+", header)
        text = token.match
        if text == "{"
            depth += 1
        elseif text == "}"
            depth -= 1
        elseif depth == 0
            # the count is the integer opening the list, not the first integer in the file
            text == "(" && return count
            count = tryparse(Int, text)
        end
    end
    return nothing
end

function _polyMesh_mismatch(polyMeshDir, mesh)
    npoints = _foam_declared_count(joinpath(polyMeshDir, "points"))
    nfaces = _foam_declared_count(joinpath(polyMeshDir, "owner"))
    if !isnothing(npoints) && npoints != length(mesh.nodes)
        return "points declares $npoints points, the mesh has $(length(mesh.nodes)) nodes"
    end
    if !isnothing(nfaces) && nfaces != length(mesh.faces)
        return "owner declares $nfaces faces, the mesh has $(length(mesh.faces)) faces"
    end
    return nothing
end

initialise_writer(format::OpenFOAM, mesh::Mesh3) = begin
    # create dummy file to load results in ParaView
    touch("XCALibre.foam")
    default_dir = "constant/polyMesh"

    mesh_files = ("points", "faces", "owner", "neighbour", "boundary")
    if all(name -> isfile(joinpath(default_dir, name)), mesh_files)
        mismatch = _polyMesh_mismatch(default_dir, mesh)
        if isnothing(mismatch)
            @info "Preserving existing mesh in constant/polyMesh."
            return FOAMWriter(nothing, nothing)
        end
        @warn "Existing constant/polyMesh does not match the simulation mesh ($mismatch). Overwriting it."
    end

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
            # Julia's shortest round-trippable representation preserves the
            # input mesh precision. `%g` only retained six significant digits.
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

    # return dummy structure for dispatch
    FOAMWriter(nothing, nothing)
end

initialise_writer(format::OpenFOAM, mesh) = error("
The OpenFOAM format can only be used for 3D simulations. Use `output=VTK()` instead.
")

function write_results(
    iteration::TI, time, mesh, meshData::FOAMWriter, BCs, args...; suffix=nothing) where TI
    timedir = ""
    if iteration == time
        timedir = @sprintf "%i" iteration
    else
        timedir = @sprintf "%.8f" time
    end

    timedirpath = mkpath(timedir)

    backend = _get_backend(mesh)
    boundaries_cpu = get_data(mesh.boundaries, backend) # get cpu copy
    labels = first.(args)

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
                write(io, _foam_dimensions(label, labels))
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
                if suffix === nothing 
                    fieldBCs = getproperty(BCs, Symbol(label))
                elseif suffix == ""
                    fieldBCs = getproperty(BCs, :p)
                end
                for BC ∈ fieldBCs
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
                write(io, _foam_dimensions(label, labels))
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
                if suffix === nothing 
                    fieldBCs = getproperty(BCs, Symbol(label))
                elseif suffix == ""
                    fieldBCs = getproperty(BCs, :U)
                end
                for BC ∈ fieldBCs
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

# OpenFOAM utilities refuse a field without the entry; pressure is kinematic unless density is written
# alongside it, which every compressible and multiphase output does; unknown names are dimensionless
function _foam_dimensions(label, labels)
    pa = "rho" ∈ labels
    d = label == "U" ? "0 1 -1" : label ∈ ("p", "p_rgh") ? (pa ? "1 -1 -2" : "0 2 -2") :
        label == "k" ? "0 2 -2" : label == "omega" ? "0 0 -1" : label ∈ ("nut", "nu") ? "0 2 -1" :
        label == "rho" ? "1 -3 0" : label == "phi" ? (pa ? "1 0 -1" : "0 3 -1") : label == "y" ? "0 1 0" : "0 0 0"
    "dimensions      [$d $(label == "T" ? "1" : "0") 0 0 0];\n"
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

_foam_boundary_entry(BC::Empty) =  begin
    """
    \t{
    \t\ttype empty;
    \t}
    """
end
