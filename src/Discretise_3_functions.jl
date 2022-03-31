# export discretise!
# export apply_boundary_conditions!
export dirichlet
# export @define_boundaries, update!
# export boundary_index
export @discretise

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
            # mesh = model.terms.term1.phi.mesh
            # cells = mesh.cells
            # faces = mesh.faces
            (; faces, cells) = mesh
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
                        $aP!
                        $aN!                    
                    # end
                end
                b[cID] = zero(0.0)
                $b!
            end
            nothing
        end # end function
    end |> esc # end quote and escape!
end # end macro

# function discretise!(equation::Equation{I,F}, model, mesh::Mesh2{I,F}) where {I,F}
#     (; faces, cells) = mesh
#     (; A, b) = equation
#     @inbounds for cID ∈ eachindex(cells)
#         cell = cells[cID]
#         A[cID,cID] = zero(0.0)
#         @inbounds for fi ∈ eachindex(cell.facesID)
#             fID = cell.facesID[fi]
#             nsign = cell.nsign[fi]
#             face = faces[fID]
#             nID = cell.neighbours[fi]   
#             A[cID,cID] += model.ap!(cell, face, nsign, cID) 
#             # Asign neighbour coefficient
#             A[cID,nID] = zero(0.0)
#             A[cID,nID] += model.an!(cell, face, nsign, cID, nID)               
#         end
#         b[cID] = zero(0.0)
#         b[cID] = model.b!(cell, cID)
#     end
# end # end function


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

function dirichlet(::Laplacian{Linear}, tsign, value)
    b_ap(cell, face) = tsign*(-J*face.area/face.delta)        # A[cellID,cellID] 
    b_b(cell, face)  = tsign*(-J*face.area/face.delta*value)  # b[cellID]
    b_ap, b_b
end

# for (faceID, cellID) ∈ zip(facesID, cellsID)
#     (; area, delta) = faces[faceID]
#     A[cellID,cellID] += (-J*area/delta)
#     b[cellID] += (-J*area/delta*left)
# end

function boundary_index(mesh::Mesh2{I,F}, name::Symbol) where {I,F}
    (; boundaries) = mesh
    for i ∈ eachindex(boundaries)
        if boundaries[i].name == name
            return i 
        end
    end
end

# macro define_boundaries(model, BCs)
#     model = esc(model);
#     BCs = esc(BCs);
#     # name = esc(:test)
#     quote
#         extract_bcs = Expr(:block)
#         for (i, bc) ∈ enumerate($BCs)
#             extract = :($(Symbol("b$i")) = BCs[$i])
#             push!(extract_bcs.args, extract)
#         end

#         assign_loop = quote
#             bID = boundary_index(mesh, b1[2])
#             boundary = boundaries[bID]
#             (; facesID, cellsID) = boundary
#             for (faceID, cellID) ∈ zip(facesID, cellsID)
#                 (; area, delta) = faces[faceID]
#                 A[cellID,cellID] += (-J*area/delta)
#                 b[cellID] += (-J*area/delta*left)
#             end
#         end
        
#         quote
#             function update!(equation, mesh, model, BCs)
#                 (; A, b) = equation
#                 (; cells, faces, boundaries) = mesh
#                 $extract_bcs
#                 $assign_loop
#             end
#         end #|> eval
#     end
#     # m = model |> eval
#     # B_ap = Expr(:block)
#     # B_b = Expr(:block)
#     # for i ∈ eachindex(m.terms)
#     #     push!(B_ap.args, :(B_aP!(
#     #         model.terms.$(Symbol("term$t")), A, cell, face, nsign, cID)
#     #         ))
#     #     push!(B_b.args, :(B_b!(
#     #         model.terms.$(Symbol("term$t")), A, cell, face, nsign, cID, nID)
#     #         ))
#     # end 
#     # quote
#     # function update!($m)
#     #     println(m.terms[1])
#     # end
#     # end |> esc
# end

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