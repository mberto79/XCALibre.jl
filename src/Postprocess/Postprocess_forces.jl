export pressure_forces, viscous_forces
export stress_tensor


pressure_forces(patch::Symbol, p::ScalarField, rho) = begin
    mesh = p.mesh
    ID = boundary_index(mesh.boundaries, patch)
    @info "calculating pressure forces on patch: $patch at index $ID"
    boundary = mesh.boundaries[ID]
    (; facesID, cellsID) = boundary
    x = FaceScalarField(zeros(Float64, length(cellsID)), mesh)
    y = FaceScalarField(zeros(Float64, length(cellsID)), mesh)
    z = FaceScalarField(zeros(Float64, length(cellsID)), mesh)
    snflux = FaceVectorField(x,y,z, mesh)
    surface_flux(snflux, facesID, cellsID, p)
    sumx, sumy, sumz = 0.0, 0.0, 0.0, 0.0
    for i ∈ eachindex(snflux.x)
        fID = facesID[i]
        sumx += snflux.x[i]
        sumy += snflux.y[i]
        sumz += snflux.z[i]
    end
    rho.*[sumx, sumy, sumz]
end

viscous_forces(patch::Symbol, U::VectorField, rho, ν, νt) = begin
    mesh = U.mesh
    faces = mesh.faces
    boundaries = mesh.boundaries
    nboundaries = length(U.BCs)
    ID = boundary_index(mesh.boundaries, patch)
    @info "calculating viscous forces on patch: $patch at index $ID"
    boundary = mesh.boundaries[ID]
    (; facesID, cellsID) = boundary
    x = FaceScalarField(zeros(Float64, length(cellsID)), mesh)
    y = FaceScalarField(zeros(Float64, length(cellsID)), mesh)
    z = FaceScalarField(zeros(Float64, length(cellsID)), mesh)
    snGrad = FaceVectorField(x,y,z, mesh)
    for i ∈ 1:nboundaries
        if ID == U.BCs[i].ID
        surface_normal_gradient(snGrad, facesID, cellsID, U, U.BCs[i].value)
        end
    end
    sumx, sumy, sumz = 0.0, 0.0, 0.0, 0.0
    for i ∈ eachindex(snGrad)
        fID = facesID[i]
        cID = cellsID[i]
        face = faces[fID]
        area = face.area
        sumx += snGrad.x[i]*area*(ν + νt[cID]) # this may need using νtf? (wall funcs)
        sumy += snGrad.y[i]*area*(ν + νt[cID])
        sumz += snGrad.z[i]*area*(ν + νt[cID])
    end
    rho.*[sumx, sumy, sumz]
end

stress_tensor(U, ν, νt) = begin
    gradU = Grad{Linear}(U)
    gradUT = T(gradU)
    Uf = FaceVectorField(U.mesh)
    grad!(gradU, Uf, U, U.BCs)
    nueff = ScalarField(U.mesh) # temp variable
    nueff.values .= ν .+ νt.values
    Reff = TensorField(U.mesh)
    for i ∈ eachindex(Reff)
        Reff[i] = -nueff[i].*(gradU[i] .+ gradUT[i])
    end
    return Reff
end

# viscous_forces(patch::Symbol, Reff::TensorField, U::VectorField, rho, ν, νt) = begin
#     mesh = U.mesh
#     faces = mesh.faces
#     ID = boundary_index(mesh.boundaries, patch)
#     @info "calculating viscous forces on patch: $patch at index $ID"
#     boundary = mesh.boundaries[ID]
#     (; facesID, cellsID) = boundary
#     x = FaceScalarField(zeros(Float64, length(cellsID)), mesh)
#     y = FaceScalarField(zeros(Float64, length(cellsID)), mesh)
#     z = FaceScalarField(zeros(Float64, length(cellsID)), mesh)
#     snGrad = FaceVectorField(x,y,z, mesh)
#     surface_flux(snGrad, facesID, cellsID, Reff)
#     # surface_normal_gradient(snGrad, facesID, cellsID, U, U.BCs[ID].value)
#     sumx, sumy, sumz = 0.0, 0.0, 0.0, 0.0
#     for i ∈ eachindex(snGrad)
#         fID = facesID[i]
#         cID = cellsID[i]
#         face = faces[fID]
#         area = face.area
#         sumx += snGrad.x[i] #*area*(ν + νt[cID]) # this may need to be using νtf?
#         sumy += snGrad.y[i] #*area*(ν + νt[cID])
#         sumz += snGrad.z[i] #*area*(ν + νt[cID])
#     end
#     rho.*[sumx, sumy, sumz]
# end