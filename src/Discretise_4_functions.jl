export apply_boundary_conditions!, boundary_conditions!, assign_boundary_conditions!
export generate_boundary_conditions!k
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
            bap!, bb! = dirichlet(term, cell, face, value)
            A[cellID,cellID] += bap!
            b[cellID] += bb!
        end 
    end
end

function generate_boundary_conditions!(
    equation::Equation{I,F}, mesh::Mesh2{I,F},model, BCs) where {I,F}

    expand = unpack(BCs)
    quote
        function update_boundaries!(
            equation::Equation{I,F}, mesh::Mesh2{I,F},model, BCs) where {I,F}
            $expand
        end
    end
end

function unpack(BCs)
    n = length(BCs)
    expand = Expr(:block)
    for i ∈ 1:n
        term = :($(Symbol(:b,i)) = BCs[$i])
        push!(expand.args, term)
    end
    return expand
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