export discretise!

discretise!(eqn, prev) = _discretise!(eqn.model, eqn, prev)

@generated function _discretise!(
    model::Model{T,S,TN,SN}, eqn, prev
    ) where {T,S,TN,SN}

    nTerms = TN
    nSources = SN

    assignment_block_1 = Expr[] # Ap
    assignment_block_2 = Expr[] # An or b
    assignment_block_3 = Expr[] # b (sources)

    # Loop for operators
    for t ∈ 1:nTerms
        function_call = quote
            scheme!(
                model.terms[$t], nzval, cell, face, cellN, ns, cIndex, nIndex, fID, prev
                )
        end
        push!(assignment_block_1, function_call)

        assign_source = quote
            scheme_source!(model.terms[$t], b, nzval, cell, cID, cIndex, prev)
        end
        push!(assignment_block_2, assign_source)
    end

    # Loop for sources
    for s ∈ 1:nSources
        add_source = quote
            (; field, sign) = model.sources[$s]
            b[cID] += sign*field[cID]*volume
        end
        push!(assignment_block_3, add_source)
    end

    quote
        (; A, b) = eqn.equation
        mesh = model.terms[1].phi.mesh
        (; faces, cells) = mesh
        (; rowval, colptr, nzval) = A
        fz = zero(Float64) # replace with func to return mesh type (Mesh module)
        @inbounds for i ∈ eachindex(nzval)
            nzval[i] = fz
        end
        cIndex = zero(Int64) # replace with func to return mesh type (Mesh module)
        nIndex = zero(Int64) # replace with func to return mesh type (Mesh module)
        @inbounds for cID ∈ eachindex(cells)
            cell = cells[cID]
            (; facesID, nsign, neighbours) = cell
            @inbounds for fi ∈ eachindex(facesID)
                fID = cell.facesID[fi]
                ns = cell.nsign[fi] # normal sign
                face = faces[fID]
                nID = cell.neighbours[fi]
                cellN = cells[nID]

                start = colptr[cID]
                offset = findfirst(isequal(cID),@view rowval[start:end]) - 1
                cIndex = start + offset

                start = colptr[nID]
                offset = findfirst(isequal(cID),@view rowval[start:end]) - 1
                nIndex = start + offset
                $(assignment_block_1...)    
            end
            b[cID] = zero(0.0)
            volume = cell.volume
            $(assignment_block_2...)
            $(assignment_block_3...)
        end
        nothing
    end
end