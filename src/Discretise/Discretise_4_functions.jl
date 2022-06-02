export generate_boundary_conditions!, update_boundaries!
export boundary_index

function generate_boundary_conditions!(mesh::Mesh2{I,F}, model, BCs) where {I,F}
    nBCs = length(BCs)
    nterms = length(model.terms)
    indices = get_boundary_indices(mesh, BCs)
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

    assignment_loops = Expr[]
    for bci ∈ 1:nBCs
        assign_loop = quote
            (; facesID, cellsID) = boundaries[$(indices[bci])]
            @inbounds for i ∈ eachindex(cellsID)
                faceID = facesID[i]
                cellID = cellsID[i]
                face = faces[faceID]; cell = cells[cellID]
            end
        end |> Base.remove_linenums!

        for ti ∈ 1:nterms
            term = Symbol(:term,ti)
            function_call = :( $(Symbol(:boundary, bci))(
                $term, A, b, cellID, cell, face, faceID
                )
            )
            push!(assign_loop.args[2].args[3].args[2].args, function_call)
        end
        push!(assignment_loops, assign_loop)
    end

    func_template = quote 
        function update_boundaries!(
            equation::Equation{I,F}, model, BCs) where {I,F}
            (; A, b, mesh) = equation
            (; boundaries, faces, cells) = mesh
            $(expand_BCs...)
            $(expand_terms...)
        end
    end |> Base.remove_linenums!

    for assignment_loop ∈ assignment_loops
        push!(func_template.args[1].args[2].args, assignment_loop.args...)
    end

    func_template |> eval
end

function get_boundary_indices(mesh::Mesh2{I,F}, BCs) where {I,F}
    BC_indices = I[]
    for BC ∈ BCs
        name = BC.name
        index = boundary_index(mesh, name)
        push!(BC_indices, index)
        println("Boundary ", name, "\t", "found at index ", index)
    end
    BC_indices
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