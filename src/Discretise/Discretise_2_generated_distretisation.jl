export discretise!

@generated function discretise!(
    equation, model::Model{T,S,TN,SN}
    ) where {T,S,TN,SN}

    nTerms = TN
    nSources = SN

    assignment_block_1 = Expr[] 
    assignment_block_2 = Expr[]

    for t ∈ 1:nTerms
        function_call = quote
            scheme!(
                model.terms[$t], nzval, cell, face, cellN, ns, cIndex, nIndex, fID
                )
        end
        push!(assignment_block_1, function_call)

        assign_source = quote
            scheme_source!(model.terms[$t], b, nzval, cell, cID)
        end
        push!(assignment_block_2, assign_source)
    end

    quote
        # (; A, b, mesh) = equation
        (; A, b) = equation
        mesh = model.terms[1].phi.mesh
        (; faces, cells) = mesh
        (; rowval, colptr, nzval) = A
        fz = zero(0.0)
        @inbounds for i ∈ eachindex(nzval)
            nzval[i] = fz
        end
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
            $(assignment_block_2...)
        end
        nothing
    end
end