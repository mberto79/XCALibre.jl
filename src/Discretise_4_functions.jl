export apply_boundary_conditions!, boundary_conditions!, assign_boundary_conditions!
export generate_boundary_conditions!, update_boundaries!
export dirichlet

function apply_boundary_conditions!(
    equation::Equation{I,F}, mesh::Mesh2{I,F}, model,
    J, left, right, bottom, top) where {I,F}
    (; boundaries, faces, cells) = mesh
    (; A, b) = equation
    for boundary ∈ boundaries
        (; facesID, cellsID) = boundary
        if boundary.name == :inlet 
            for (faceID, cellID) ∈ zip(facesID, cellsID)
                (; area, delta) = faces[faceID]
                A[cellID,cellID] += (-J*area/delta)
                b[cellID] += (-J*area/delta*left)
            end
        elseif boundary.name == :outlet 
            for (faceID, cellID) ∈ zip(facesID, cellsID)
                (; area, delta) = faces[faceID]
                A[cellID,cellID] += (-J*area/delta)
                b[cellID] += (-J*area/delta*right)
            end 
        elseif boundary.name == :bottom 
            for (faceID, cellID) ∈ zip(facesID, cellsID)
                (; area, delta) = faces[faceID]
                A[cellID,cellID] += (-J*area/delta)
                b[cellID] += (-J*area/delta*bottom)
            end
        elseif boundary.name == :top 
            for (faceID, cellID) ∈ zip(facesID, cellsID)
                (; area, delta) = faces[faceID]
                A[cellID,cellID] += (-J*area/delta)
                b[cellID] += (-J*area/delta*top)
            end
        end
    end
end

function boundary_conditions!(
    equation::Equation{I,F} , mesh::Mesh2{I,F}, J, BCs) where {I,F}
    (; boundaries, faces, cells) = mesh
    (; A, b) = equation
    for BC ∈ BCs
        bID = boundary_index(mesh, BC[2])
        (; facesID, cellsID) = boundaries[bID]
        for (faceID, cellID) ∈ zip(facesID, cellsID)
            (; area, delta) = faces[faceID]
            A[cellID,cellID] += (-J*area/delta)
            b[cellID] += (-J*area/delta*BC[3])
        end
    end
end

function assign_boundary_conditions!(
    equation::Equation{I,F}, mesh::Mesh2{I,F},model, BCs) where {I,F}
    (; boundaries, faces, cells) = mesh
    (; A, b) = equation
    term = model.terms.term1
    for BC ∈ BCs 
        name = BC[2]
        value = BC[3]
        bID = boundary_index(mesh, name)
        (; facesID, cellsID) = boundaries[bID]
        # @time for (faceID, cellID) ∈ zip(facesID, cellsID)
        for i ∈ eachindex(cellsID)
            faceID = facesID[i]
            cellID = cellsID[i]
            face = faces[faceID]; cell = cells[cellID]
            # ap!, b! = 
            dirichlet(term, A, b, cellID, cell, face, value)
            # A[cellID,cellID] += ap!
            # b[cellID] += b!
        end 
    end
end

function generate_boundary_conditions!(
    equation::Equation{I,F}, mesh::Mesh2{I,F},model, BCs) where {I,F}

    nBCs = length(BCs)
    nterms = length(model.terms)

    expand_BCs = Expr[]
    for i ∈ 1:nBCs
        boundary_conditions = :($(Symbol(:boundary,i)) = BCs[$i])
        push!(expand_BCs, boundary_conditions)
    end
    expand_terms = Expr[]
    for i ∈ 1:nterms
        model_terms = :($(Symbol(:term,i)) = model.terms.$(Symbol(:term,i)))
        push!(expand_terms, model_terms)
    end

    assignment_loops = Expr[] #Expr(:block)
    # expand_function_call = Expr[]#Expr(:block)
    for bci ∈ 1:nBCs
        assign_loop = quote
            bID = boundary_index(mesh, $(Symbol(:boundary, bci))[2])
            (; facesID, cellsID) = boundaries[bID]
            @inbounds for i ∈ eachindex(cellsID)
                faceID = facesID[i]
                cellID = cellsID[i]
                face = faces[faceID]; cell = cells[cellID]
                # $(expand_function_call...)
            end
        end |> Base.remove_linenums!

        for ti ∈ 1:nterms
            term = Symbol(:term,ti)
            function_call = :( $(Symbol(:boundary, bci))[1](
                $term, A, b, cellID, cell, face, $(Symbol(:boundary, bci))[3]
                )
            )
            push!(assign_loop.args[3].args[3].args[2].args, function_call)
        end
        push!(assignment_loops, assign_loop)
    end

    func_template = quote 
        function update_boundaries!(
            equation::Equation{I,F}, mesh::Mesh2{I,F},model, BCs) where {I,F}
            (; boundaries, faces, cells) = mesh
            (; A, b) = equation
            $(expand_BCs...)
            $(expand_terms...)
        end
    end |> Base.remove_linenums!

    for assignment_loop ∈ assignment_loops
        push!(func_template.args[1].args[2].args, assignment_loop.args...)
    end

    func_template |> eval
end


function boundary_index(mesh::Mesh2{I,F}, name::Symbol) where {I,F}
    (; boundaries) = mesh
    for i ∈ eachindex(boundaries)
        if boundaries[i].name == name
            return i 
        end
    end
end

# # begin
# #     J = model.terms.term1.J
# #     sgn = model.terms.term1.sign[1]

# #     area, delta, normal, nsign = face_properties(mesh, 1)
# #     b[1] += sgn*(J⋅(area*normal*nsign))*leftBC

# #     area, delta, normal, nsign = face_properties(mesh, nCells+1)
# #     b[nCells] += sgn*(J⋅(area*normal*nsign))*rightBC
# # end

# # begin
# #     J = model.terms.term2.J
# #     sgn = model.terms.term2.sign[1]

# #     area, delta, normal, nsign = face_properties(mesh, 1)
# #     b[1] += sgn*(-J*area/delta*leftBC)
# #     A[1,1] += sgn*(-J*area/delta)

# #     area, delta, normal, nsign = face_properties(mesh, nCells+1)
# #     b[nCells] += sgn*(-J*area/delta*rightBC)
# #     A[nCells,nCells] += sgn*(-J*area/delta)

# # end