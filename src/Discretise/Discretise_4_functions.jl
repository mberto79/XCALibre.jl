export apply_boundary_conditions!
export boundary_index
export H!

@generated function apply_boundary_conditions!(
    equation::Equation{I,F}, model, BCs) where {I,F}

    # Unpack terms that make up the model (not sources)
    terms = Expr[]
    for term ∈ model.types[1].parameters[1]
        term_extracted = :($term = model.terms.$term)
        push!(terms, term_extracted)
    end

    # Definition of main assignment loop (one per patch)
    assignment_loops = []
    for bci ∈ 1:length(BCs.parameters)
        func_calls = Expr[]
        for term ∈ model.types[1].parameters[1] 
            # call = Expr(:call, :(BCs[$bci]), term, :A, :b, :cellID, :cell, :face, :faceID)
            call = quote
                (BCs[$bci])($term, A, b, cellID, cell, face, faceID)
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
    indices = get_boundary_indices(mesh, BCs)
    $(terms...)
    $(assignment_loops...)
    nothing
    end
end

function get_boundary_indices(mesh::Mesh2{TI,TF}, BCs) where {TI,TF}
    BC_indices = TI[]
    # BC_indices = ()
    for BC ∈ BCs
        name = BC.name
        index = boundary_index(mesh, name)
        push!(BC_indices, index)
        # BC_indices = (BC_indices..., index)
        # println("Boundary ", name, "\t", "found at index ", index)
    end
    BC_indices
end

function boundary_index(mesh::Mesh2{TI,TF}, name::Symbol) where {TI,TF}
    (; boundaries) = mesh
    bci = zero(TI)
    for i ∈ eachindex(boundaries)
        bci += 1
        if boundaries[i].name == name
            return bci 
        end
    end
end

function H!(Hv::VectorField, v::VectorField{I,F}, xeqn, yeqn) where {I,F}
    (; x, y, z, mesh) = Hv 
    (; cells, faces) = mesh
    Ax = xeqn.A;  Ay = yeqn.A
    bx = xeqn.b; by = yeqn.b; # bz = zeros(length(bx))
    
    @inbounds for cID ∈ eachindex(cells)
        cell = cells[cID]
        (; neighbours) = cell
        sumx = zero(F)
        sumy = zero(F)
        @inbounds for nID ∈ neighbours
            sumx += Ax[cID,nID]*v.x[nID]
            sumy += Ay[cID,nID]*v.y[nID]
        end
        rD = 1.0/Ax[cID, cID]
        x[cID] = (bx[cID] - sumx)*rD
        y[cID] = (by[cID] - sumy)*rD
        z[cID] = zero(F)
    end
end