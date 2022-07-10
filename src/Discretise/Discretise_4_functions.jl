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
    indices = boundary_indices(mesh, BCs)
    $(terms...)
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

function H!(Hv::VectorField, v::VectorField{I,F}, xeqn, yeqn) where {I,F}
    (; x, y, z, mesh) = Hv 
    (; cells, faces) = mesh
    Ax = xeqn.A;  Ay = yeqn.A
    bx = xeqn.b; by = yeqn.b; # bz = zeros(length(bx))
    
    vx, vy = v.x, v.y
    @inbounds for cID ∈ eachindex(cells)
        cell = cells[cID]
        (; neighbours) = cell
        sumx = zero(F)
        sumy = zero(F)
        @inbounds for nID ∈ neighbours
            sumx += Ax[cID,nID]*vx[nID]
            sumy += Ay[cID,nID]*vy[nID]
        end
        rD = 1.0/Ax[cID, cID]
        x[cID] = (bx[cID] - sumx)*rD
        y[cID] = (by[cID] - sumy)*rD
        z[cID] = zero(F)
    end
end

function H!(
    Hv::VectorField, ux::ScalarField{I,F}, uy::ScalarField{I,F}, xeqn, yeqn
    ) where {I,F}
    (; x, y, z, mesh) = Hv 
    (; cells, faces) = mesh
    Ax = xeqn.A;  Ay = yeqn.A
    bx = xeqn.b; by = yeqn.b; # bz = zeros(length(bx))
    ux_vals = ux.values
    uy_vals = uy.values
    
    @inbounds for cID ∈ eachindex(cells)
        cell = cells[cID]
        (; neighbours) = cell
        sumx = zero(F)
        sumy = zero(F)
        @inbounds for nID ∈ neighbours
            sumx += Ax[cID,nID]*ux_vals[nID]
            sumy += Ay[cID,nID]*uy_vals[nID]
        end
        rD = 1.0/Ax[cID, cID]
        x[cID] = (bx[cID] - sumx)*rD
        y[cID] = (by[cID] - sumy)*rD
        z[cID] = zero(F)
    end
end