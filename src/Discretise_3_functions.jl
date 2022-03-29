export discretise!
export apply_boundary_conditions!

function discretise!(equation::Equation{I,F}, model, mesh::Mesh2{I,F}) where {I,F}
    # mesh = model.terms.term1.ϕ.mesh
    (; faces, cells) = mesh
    # cells = mesh.cells
    # faces = mesh.faces
    (; A, b) = equation
    # A = equation.A
    # b = equation.b
    @inbounds for cID ∈ eachindex(cells)
        cell = cells[cID]
        A[cID,cID] = zero(0.0)
        @inbounds for fi ∈ eachindex(cell.facesID)
            fID = cell.facesID[fi]
            nsign = cell.nsign[fi]
            face = faces[fID]
            nID = cell.neighbours[fi]
            # c1 = face.ownerCells[1]
            # c2 = face.ownerCells[2]
            # if c1 != c2 
                A[cID,nID] = zero(0.0)
                # $aP!
                # $aN!     
                A[cID,cID] += model.ap!(cell, face, nsign, cID)               
                A[cID,nID] += model.an!(cell, face, nsign, cID, nID)               
            # end
        end
        b[cID] = zero(0.0)
        # $b!
        b[cID] = model.b!(cell, cID)
    end
end # end function


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

# begin
#     J = model.terms.term1.J
#     sgn = model.terms.term1.sign[1]

#     area, delta, normal, nsign = face_properties(mesh, 1)
#     b[1] += sgn*(J⋅(area*normal*nsign))*leftBC

#     area, delta, normal, nsign = face_properties(mesh, nCells+1)
#     b[nCells] += sgn*(J⋅(area*normal*nsign))*rightBC
# end

# begin
#     J = model.terms.term2.J
#     sgn = model.terms.term2.sign[1]

#     area, delta, normal, nsign = face_properties(mesh, 1)
#     b[1] += sgn*(-J*area/delta*leftBC)
#     A[1,1] += sgn*(-J*area/delta)

#     area, delta, normal, nsign = face_properties(mesh, nCells+1)
#     b[nCells] += sgn*(-J*area/delta*rightBC)
#     A[nCells,nCells] += sgn*(-J*area/delta)

# end