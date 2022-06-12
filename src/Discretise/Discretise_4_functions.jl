export generate_boundary_conditions!, update_boundaries!
export boundary_index
export H!

function generate_boundary_conditions!(name, mesh::Mesh2{I,F}, model, BCs) where {I,F}
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
        # function update_boundaries!(
        function $name(
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

function H!(Hv::VectorField, v::VectorField{I,F}, xeqn, yeqn, B, V, H) where {I,F}
    (; x, y, z) = Hv 
    Ax = xeqn.A;  Ay = yeqn.A
    bx = xeqn.b; by = yeqn.b; # bz = zeros(length(bx))

    D = @view Ax[diagind(Ax)]
    Di = Diagonal(D)

    # B = [bx by bz]
    B[:,1] .= bx
    B[:,2] .= by 
    # B[:,3] .= bz

    V[:,1] .= v.x
    V[:,2] .= v.y

    # H = ( B .- (Ax .- Di) * V )./D
    H .= ( B .- (Ax .- Di) * V )./D
    
    x .= @view H[:,1]
    y .= @view H[:,2]
    z .= @view H[:,3]
    nothing
end