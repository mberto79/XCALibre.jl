export stress_tensor, wall_shear_stress
export pressure_force, viscous_force
export lift_to_drag, aero_coeffs
export paraview_vis, y_plus

paraview_vis(;paraview_path::String,vtk_path::String) = begin
    try
        run(`"$paraview_path" "$vtk_path"`)
    catch
        println("\nCannot open paraview: check installation location & integrity!")
    end
end

y_plus(patch::Symbol, ρ::Float64, model::RANS{M,F1,F2,V,T,E,D}) where {M,F1,F2,V,T,E,D} = begin
    M == Laminar ? nut = ConstantScalar(0.0) : nut = model.turbulence.nut
    tau, face_centres = wall_shear_stress(patch, model)
    (; mesh, nu) = model
    (; boundaries, cells,faces) = mesh
    ID = boundary_index(boundaries, patch)
    boundary = boundaries[ID]
    (; cellsID) = boundary
    @info "calculating y+ values on patch: $patch at index $ID"
    sumX,sumY,sumZ = 0.0,0.0,0.0
    for i ∈ eachindex(tau)
            sumX += tau.x[i]
            sumY += tau.y[i]
            sumZ += tau.z[i]
    end
    yplus = Array{Float64,1}(undef,length(face_centres))
    y = Array{Float64,1}(undef,length(face_centres))
    for (i,face_centre) ∈ enumerate(face_centres)
        ustar = √(norm([sumX,sumY,sumZ])/ρ)
        y[i] = 2*norm(face_centre-cells[cellsID[i]].centre) #First cell height = 2*distance from boundary face to first cell centre
        yplus[i] = (ustar*y[i])/(nu[cellsID[i]]+nut[cellsID[i]])
    end
    print("\nAverage y+ value on patch: ",round(mean(yplus),sigdigits = 4))
    return yplus,y
end

lift_to_drag(patch::Symbol, ρ, model::RANS{M,F1,F2,V,T,E,D}) where {M,F1,F2,V,T,E,D} = begin
    oldstd = stdout
    redirect_stdout(devnull)
    Fp = pressure_force(patch, ρ, model)
    Fv = viscous_force(patch, ρ, model)
    redirect_stdout(oldstd)
    Ft = Fp + Fv
    aero_eff = Ft[2]/Ft[1]
    print("Aerofoil L/D: ",round(aero_eff,sigdigits = 4))
    return aero_eff
end

aero_coeffs(patch::Symbol, chord::R where R <: Real, ρ, velocity::Vector{Float64}, model::RANS{M,F1,F2,V,T,E,D}) where {M,F1,F2,V,T,E,D} = begin
    oldstd = stdout
    redirect_stdout(devnull)
    Fp = pressure_force(patch, ρ, model)
    Fv = viscous_force(patch, ρ, model)
    redirect_stdout(oldstd)
    Ft = Fp + Fv
    C_l = 2Ft[2]/(ρ*(velocity[1]^2)*chord*0.001)
    C_d = 2Ft[1]/(ρ*(velocity[1]^2)*chord*0.001)
    print("Lift Coefficient: ",round(C_l,sigdigits = 4))
    print("\nDrag Coefficient: ",round(C_d,sigdigits = 4))
    return C_l,C_d
end

pressure_force(patch::Symbol, rho, model::RANS{M,F1,F2,V,T,E,D}) where {M,F1,F2,V,T,E,D} = begin
    (; mesh, p) = model
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
    Fp = rho.*[sumx, sumy, sumz]
    print("\nPressure force: (", Fp[1], " ", Fp[2], " ", Fp[3], ")\n")
    return Fp
end

viscous_force(patch::Symbol, rho, model::RANS{M,F1,F2,V,T,E,D}) where {M,F1,F2,V,T,E,D} = begin
    M == Laminar ? nut = ConstantScalar(0.0) : nut = model.turbulence.nut
    (; mesh, U, nu) = model
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
        sumx += snGrad.x[i]*area*(nu[cID] + nut[cID]) # this may need using νtf? (wall funcs)
        sumy += snGrad.y[i]*area*(nu[cID] + nut[cID])
        sumz += snGrad.z[i]*area*(nu[cID] + nut[cID])
    end
    Fv = rho.*[sumx, sumy, sumz]
    print("\nViscous force: (", Fv[1], " ", Fv[2], " ", Fv[3], ")\n")
    return Fv
end

wall_shear_stress(patch::Symbol, model::RANS{M,F1,F2,V,T,E,D}) where {M,F1,F2,V,T,E,D} = begin
    # Line below needs to change to do selection based on nut BC
    M == Laminar ? nut = ConstantScalar(0.0) : nut = model.turbulence.nut
    (; mesh, U, nu) = model
    (; boundaries, faces) = mesh
    ID = boundary_index(boundaries, patch)
    boundary = boundaries[ID]
    (; facesID, cellsID) = boundary
    @info "calculating viscous forces on patch: $patch at index $ID"
    x = FaceScalarField(zeros(Float64, length(cellsID)), mesh)
    y = FaceScalarField(zeros(Float64, length(cellsID)), mesh)
    z = FaceScalarField(zeros(Float64, length(cellsID)), mesh)
    tauw = FaceVectorField(x,y,z, mesh)
    Uw = zero(_get_float(mesh))
    for i ∈ 1:length(U.BCs)
        if ID == U.BCs[i].ID
            Uw = U.BCs[i].value
        end
    end
    surface_normal_gradient(tauw, facesID, cellsID, U, Uw)
    pos = fill(SVector{3,Float64}(0,0,0), length(facesID))
    for i ∈ eachindex(tauw)
        fID = facesID[i]
        cID = cellsID[i]
        face = faces[fID]
        nueff = nu[cID]  + nut[cID]
        tauw.x[i] *= nueff # this may need using νtf? (wall funcs)
        tauw.y[i] *= nueff
        tauw.z[i] *= nueff
        pos[i] = face.centre
    end
    
    return tauw, pos
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