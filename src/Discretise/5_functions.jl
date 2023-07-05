export apply_boundary_conditions!
export boundary_index
export assign

function assign(field, BCs...)
    # use this function to pack boundary ID info at runtime
    nothing
end

function apply_boundary_conditions!(equation, model, BCs)
    (; mesh) = equation
    indices = boundary_indices(mesh, BCs)
    update_boundary_conditions!(equation, model, BCs, indices)
end

@generated function update_boundary_conditions!(
    equation::Equation{I,F}, model, BCs, indices) where {I,F}

    # Unpack terms that make up the model (not sources)
    # terms = Expr[]
    nTerms = model.parameters[3]
    # for t ∈ 1:nTerms
    #     term_extracted = :(term$t = model.terms[$t])
    #     push!(terms, term_extracted)
    # end

    # Definition of main assignment loop (one per patch)
    assignment_loops = []
    for bci ∈ 1:length(BCs.parameters)
        func_calls = Expr[]
        for t ∈ 1:nTerms 
            # call = Expr(:call, :(BCs[$bci]), term, :A, :b, :cellID, :cell, :face, :faceID)
            call = quote
                # (BCs[$bci])(term$t, A, b, cellID, cell, face, faceID)
                (BCs[$bci])(model.terms[$t], A, b, cellID, cell, face, faceID)
            end
            push!(func_calls, call)
        end
        assignment_loop = quote
            (; facesID, cellsID) = boundaries[indices[$bci]]
            @inbounds for i ∈ eachindex(cellsID)
                faceID = facesID[i]
                cellID = cellsID[i]
                face = faces[faceID]
                cell = cells[cellID]
                $(func_calls...)
            end
        end
        push!(assignment_loops, assignment_loop.args...)
    end

    quote
    (; A, b, mesh) = equation
    (; boundaries, faces, cells) = mesh
    # indices = boundary_indices(mesh, BCs)
    # $(terms...)
    $(assignment_loops...)
    nothing
    end
end

@generated function boundary_indices(mesh::Mesh2{TI,TF}, BCs) where {TI,TF}
    unpacked_BCs = []
    for i ∈ 1:length(BCs.parameters)
        unpack = quote
            name = BCs[$i].name
            index = boundary_index(boundaries, name)
            BC_indices = (BC_indices..., index)
        end
        push!(unpacked_BCs, unpack)
    end
    quote
        boundaries = mesh.boundaries
        BC_indices = ()
        $(unpacked_BCs...)
        return BC_indices
    end
end

function boundary_index(boundaries::Vector{Boundary{TI}}, name::Symbol) where {TI}
    bci = zero(TI)
    for i ∈ eachindex(boundaries)
        bci += 1
        if boundaries[i].name == name
            return bci 
        end
    end
end