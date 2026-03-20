export read_UNV3

#= Single-pass state-machine parser for UNV mesh files generated in SALOME.
Utilises a pre-allocated token buffer and lazy iteration (`eachsplit`) to completely 
eliminate intermediate array and string allocations during line parsing. =#
function read_UNV3(unv_mesh; scale=1.0, integer::Type{I}=Int64, float::Type{F}=Float64) where {I, F}
    scale_F = F(scale)
    
    # 1D Concrete array initialization
    points = Point{F, SVector{3, F}}[]
    faces = Face{I, Vector{I}}[]
    cells = Cell_UNV{I, Vector{I}}[]
    boundaryElements = BoundaryElement{String, I, Vector{I}}[]
    
    # Global element tracking variables used to replicate baseline ID offsets
    edge_counter = I(0)
    face_counter = I(0)
    cell_counter = I(0)
    currentBoundary = I(0)
    
    # Pre-allocated memory buffer to hold line tokens (prevents array allocation from split)
    sline_buffer = Vector{SubString{String}}(undef, 32)
    
    @info "Loading UNV file (Parsing ASCII)..."


    open(unv_mesh, "r") do io
        dataset_id = ""
        
        while !eof(io)
            line = strip(readline(io))
            if isempty(line); continue; end
            
            # UNV State Machine: "-1" toggles data blocks open and closed
            if line == "-1"
                if dataset_id == ""
                    # Open a new block
                    dataset_id = strip(readline(io))
                    if dataset_id == "-1"; dataset_id = ""; end
                else
                    # Close the current block
                    dataset_id = ""
                end
                continue
            end
            
            # Zero-allocation token extraction into the reusable buffer
            n_tokens = 0
            for t in eachsplit(line, keepempty=false)
                n_tokens += 1
                sline_buffer[n_tokens] = t
            end
            
            # ==========================================
            # BLOCK 2411: NODES
            # ==========================================
            if dataset_id == "2411"
                # Coordinates always appear on lines with exactly 3 parameters
                if n_tokens == 3
                    x = _parse_unv_float(F, sline_buffer[1]) * scale_F
                    y = _parse_unv_float(F, sline_buffer[2]) * scale_F
                    z = _parse_unv_float(F, sline_buffer[3]) * scale_F
                    push!(points, Point(SVector{3, F}(x, y, z)))
                end
                
            # ==========================================
            # BLOCK 2412: ELEMENTS (Edges, Faces, Cells)
            # ==========================================
            elseif dataset_id == "2412"
                # Element headers always have exactly 6 parameters
                if n_tokens == 6
                    element_id = parse(I, sline_buffer[1])
                    elem_type  = parse(I, sline_buffer[2])
                    num_nodes  = parse(I, sline_buffer[6]) 
                    
                    # 1. Edges (Counted strictly for global ID offsetting)
                    if num_nodes == 2 || elem_type == 11
                        edge_counter += 1
                        readline(io) # Consume the edge nodes line
                        
                    # 2. Triangles (41) and Quadrilaterals (44)
                    elseif (elem_type == 41 && num_nodes == 3) || (elem_type == 44 && num_nodes == 4)
                        face_counter += 1
                        if element_id - edge_counter != face_counter
                            throw(ArgumentError("Face Index in UNV file are not in order! At UNV index = $element_id"))
                        end
                        
                        # Pre-allocate exact node array to prevent dynamic push! overhead
                        f_nodes = Vector{I}(undef, num_nodes)
                        idx = 1
                        for n in eachsplit(readline(io), keepempty=false)
                            f_nodes[idx] = parse(I, n)
                            idx += 1
                        end
                        push!(faces, Face(face_counter, I(num_nodes), f_nodes))
                        
                    # 3. Cells: Tetrahedra (111), Hexahedra (115), Prisms/Wedges (112)
                    elseif elem_type == 111 || elem_type == 115 || elem_type == 112
                        cell_counter += 1
                        if element_id - face_counter - edge_counter != cell_counter
                            throw(ArgumentError("Cell Index in UNV file are not in order! At UNV index = $element_id"))
                        end
                        
                        # Exact pre-allocation with robust wrapped-line tracking
                        c_nodes = Vector{I}(undef, num_nodes)
                        idx = 1
                        while idx <= num_nodes
                            for n in eachsplit(readline(io), keepempty=false)
                                c_nodes[idx] = parse(I, n)
                                idx += 1
                            end
                        end
                        push!(cells, Cell_UNV(cell_counter, I(num_nodes), c_nodes))
                    end
                end
                
            # ==========================================
            # BLOCK 2467: GROUPS / BOUNDARIES
            # ==========================================
            elseif dataset_id == "2467"
                # Group Name detected (single non-numeric string)
                if n_tokens == 1 && tryparse(I, sline_buffer[1]) === nothing
                    currentBoundary += 1
                    new_boundary = BoundaryElement(I(0))
                    new_boundary.index = I(currentBoundary)
                    new_boundary.name = String(sline_buffer[1])
                    push!(boundaryElements, new_boundary)
                    
                # Entity tracking records (Sets of 8 or 4 parameters)
                elseif currentBoundary > 0
                    if n_tokens == 8 && parse(I, sline_buffer[2]) != 0
                        push!(boundaryElements[currentBoundary].facesID, parse(I, sline_buffer[2]) - edge_counter)
                        push!(boundaryElements[currentBoundary].facesID, parse(I, sline_buffer[6]) - edge_counter)
                    elseif n_tokens == 4 && parse(I, sline_buffer[2]) != 0
                        push!(boundaryElements[currentBoundary].facesID, parse(I, sline_buffer[2]) - edge_counter)
                    end
                end
            end
        end
    end
    
    return points, faces, cells, boundaryElements
end

# ==============================================================================
# UTILITY AND AUXILIARY FUNCTIONS
# ==============================================================================

# Zero-allocation wrapper for float parsing. 
# Only triggers the string-replacing allocation if legacy FORTRAN 'D' notation is strictly detected.
@inline function _parse_unv_float(::Type{F}, s::AbstractString) where F
    if occursin('D', s) || occursin('d', s)
        return parse(F, replace(replace(s, 'D' => 'e'), 'd' => 'e'))
    end
    return parse(F, s)
end