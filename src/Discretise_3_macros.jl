export @discretise
export @discretise2
export @discretise3
export @discretise4


macro discretise(Model_type, nTerms::Integer, nSources::Integer)
    aP! = Expr(:block)
    aN! = Expr(:block)
    b!  = Expr(:block)
    for t ∈ 1:nTerms
        push!(aP!.args, :(aP!(
            model.terms.$(Symbol("term$t")), A, cell, face, nsign, cID)
            ))
        push!(aN!.args, :(aN!(
            model.terms.$(Symbol("term$t")), A, cell, face, nsign, cID, nID)
            ))
        push!(b!.args, :(
            b!(model.terms.$(Symbol("term$t")), b, cell, cID)
            ))
    end 
    
    quote 
        function discretise!(equation, model::$Model_type, mesh)
            (; faces, cells) = mesh
            (; A, b) = equation
            @inbounds for cID ∈ eachindex(cells)
                cell = cells[cID]
                A[cID,cID] = zero(0.0)
                @inbounds for fi ∈ eachindex(cell.facesID)
                    fID = cell.facesID[fi]
                    nsign = cell.nsign[fi]
                    face = faces[fID]
                    nID = cell.neighbours[fi]
                    A[cID,nID] = zero(0.0)
                    $aP!
                    $aN!                    
                end
                b[cID] = zero(0.0)
                $b!
            end
            nothing
        end # end function
    end |> esc # end quote and escape!
end # end macro

macro discretise2(Model_type, nTerms::Integer, nSources::Integer)
    assignment_block_1 = [] #Expr(:block)
    assignment_block_2 = [] #Expr(:block)
    for t ∈ 1:nTerms
        function_call = :(
            scheme!(model.terms.$(Symbol("term$t")), A, cell, face, ns, cID, nID)
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
        function discretise2!(equation, model::$Model_type, mesh)
            (; faces, cells) = mesh
            (; A, b) = equation
            @inbounds for cID ∈ eachindex(cells)
                cell = cells[cID]
                (; facesID, nsign, neighbours) = cell
                A[cID,cID] = zero(0.0)
                @inbounds for fi ∈ eachindex(cell.facesID)
                    fID = cell.facesID[fi]
                    ns = cell.nsign[fi] # normal sign
                    face = faces[fID]
                    nID = cell.neighbours[fi]
                    A[cID,nID] = zero(0.0)
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

macro discretise3(Model_type, nTerms::Integer, nSources::Integer)
    assignment_block = [] #Expr(:block)
    for t ∈ 1:nTerms
        term = Symbol("term$t")
        function_call = :(
            scheme3!(model.terms.$term, A, b, face, cells, fID, cID1, cID2)
            )
        push!(assignment_block, function_call)
    end 
    
    func = quote 
        function discretise3!(equation, model::$Model_type, mesh)
            (; faces, cells) = mesh
            (; A, b) = equation
            @inbounds for i ∈ eachindex(A.nzval)
                A.nzval[i] = 0.0
            end
            @inbounds for i ∈ eachindex(b)
                b[i] = 0.0
            end
            bfaces = total_boundary_faces(mesh)
            start = bfaces + 1
            finish = length(faces)
            @inbounds for fID ∈ start:finish
                face = faces[fID]
                (; ownerCells) = face
                cID1 = ownerCells[1]
                cID2 = ownerCells[2]
                $(assignment_block...)
            end
        end # end function
    end |> esc # end quote and escape!
    return func
end # end macro

macro discretise4(Model_type, nTerms::Integer, nSources::Integer)
    assignment_block_1 = [] #Expr(:block)
    assignment_block_2 = [] #Expr(:block)
    for t ∈ 1:nTerms
        function_call = :(
            scheme4!(model.terms.$(Symbol("term$t")), nzval, cell, face, ns, cIndex, nIndex)
            )
        # ap_assignment = :(A[cID, cID] += coeffs[1])
        # an_assignment = :(A[cID, nID] += coeffs[2])
        push!(assignment_block_1, function_call)

        assign_source = :(
            scheme_source4!(model.terms.$(Symbol("term$t")), b, cell, cID)
            )
        push!(assignment_block_2, assign_source)
    end 
    
    func = quote 
        function discretise4!(equation, model::$Model_type, mesh)
            (; faces, cells) = mesh
            (; A, b) = equation
            (; rowval, colptr, nzval) = A
            fz = zero(0.0)
            @inbounds for i ∈ eachindex(nzval)
                nzval[i] = fz
            end
            @inbounds for cID ∈ eachindex(cells)
                cell = cells[cID]
                (; facesID, nsign, neighbours) = cell
                @inbounds for fi ∈ eachindex(cell.facesID)
                    fID = cell.facesID[fi]
                    ns = cell.nsign[fi] # normal sign
                    face = faces[fID]
                    nID = cell.neighbours[fi]

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