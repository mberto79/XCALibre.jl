export @discretise
export discretise!

@generated function discretise!()
    nothing
end

macro discretise(Model_type, nTerms::Integer, nSources::Integer)
    assignment_block_1 = [] #Expr(:block)
    assignment_block_2 = [] #Expr(:block)
    for t ∈ 1:nTerms
        function_call = :(
            scheme!(model.terms.$(Symbol("term$t")), nzval, cell, face, cellN, ns, cIndex, nIndex, fID)
            )
        # ap_assignment = :(A[cID, cID] += coeffs[1])
        # an_assignment = :(A[cID, nID] += coeffs[2])
        push!(assignment_block_1, function_call)

        assign_source = :(
            scheme_source!(model.terms.$(Symbol("term$t")), b, cell, cID)
            )
        push!(assignment_block_2, assign_source)
    end 
    
    func = quote 
        function discretise!(equation, model::$Model_type)
            (; A, b, mesh) = equation
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
        end # end function
    end |> esc # end quote and escape!
    return func
end # end macro