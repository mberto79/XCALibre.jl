export boundary_average
export pressure_force, viscous_force
export stress_tensor, wall_shear_stress

"""
    pressure_force(patch::Symbol, p::ScalarField, rho)

Function to calculate the pressure force acting on a given patch/boundary.

# Input arguments

- `patch::Symbol` name of the boundary of interest (as a `Symbol`)
- `p::ScalarField` pressure field
- `rho` density. Set to 1 for incompressible solvers
"""
pressure_force(patch::Symbol, p::ScalarField, rho) = begin
    mesh = p.mesh
    ID = boundary_index(mesh.boundaries, patch)
    @info "calculating pressure forces on patch: $patch at index $ID"
    boundary = mesh.boundaries[ID]
    (; IDs_range) = boundary
    x = FaceScalarField(zeros(Float64, length(IDs_range)), mesh)
    y = FaceScalarField(zeros(Float64, length(IDs_range)), mesh)
    z = FaceScalarField(zeros(Float64, length(IDs_range)), mesh)
    snflux = FaceVectorField(x,y,z, mesh)
    surface_flux!(snflux, p, IDs_range)
    sumx, sumy, sumz = 0.0, 0.0, 0.0, 0.0
    for i ∈ eachindex(snflux.x)
        sumx += snflux.x[i]
        sumy += snflux.y[i]
        sumz += snflux.z[i]
    end
    Fp = rho.*[sumx, sumy, sumz]
    print("\n Pressure force: (", Fp[1], " ", Fp[2], " ", Fp[3], ")\n")
    return Fp
end

"""
    viscous_force(patch::Symbol, U::VectorField, rho, ν, νt)

Function to calculate the pressure force acting on a given patch/boundary.

# Input arguments

- `patch::Symbol` name of the boundary of interest (as a `Symbol`)
- `U::VectorField` pressure field
- `rho` density. Set to 1 for incompressible solvers
- `ν` laminar viscosity of the fluid
- `νt` eddy viscosity from turbulence models. Pass ConstantScalar(0) for laminar flows
"""
viscous_force(patch::Symbol, U::VectorField, rho, ν, νt, UBCs) = begin
    mesh = U.mesh
    (; faces, boundaries, boundary_cellsID) = mesh
    nboundaries = length(boundaries)
    ID = boundary_index(boundaries, patch)
    @info "calculating viscous forces on patch: $patch at index $ID"
    boundary = mesh.boundaries[ID]
    # (; facesID, cellsID) = boundary
    (; IDs_range) = boundary
    x = FaceScalarField(zeros(Float64, length(IDs_range)), mesh)
    y = FaceScalarField(zeros(Float64, length(IDs_range)), mesh)
    z = FaceScalarField(zeros(Float64, length(IDs_range)), mesh)
    snGrad = FaceVectorField(x,y,z, mesh)
    for i ∈ 1:nboundaries
        if ID == UBCs[i].ID
        surface_normal_gradient!(snGrad, U, UBCs[i].value, IDs_range)
        
        end
    end
    sumx, sumy, sumz = 0.0, 0.0, 0.0, 0.0
    for i ∈ eachindex(snGrad)
        fID = IDs_range[i]
        cID = boundary_cellsID[fID]
        face = faces[fID]
        area = face.area
        sumx += snGrad.x[i]*area*(ν + νt[cID]) # this may need using νtf? (wall funcs)
        sumy += snGrad.y[i]*area*(ν + νt[cID])
        sumz += snGrad.z[i]*area*(ν + νt[cID])
    end
    Fv = rho.*[sumx, sumy, sumz]
    print("\n Viscous force: (", Fv[1], " ", Fv[2], " ", Fv[3], ")\n")
    return Fv
end

"""
    function boundary_average(patch::Symbol, field, config; time=0)
        # Extract mesh object
        mesh = field.mesh

        # Determine ID (index) of the boundary patch 
        ID = boundary_index(mesh.boundaries, patch)
        @info "calculating average on patch: \$patch at index \$ID"
        boundary = mesh.boundaries[ID]
        (; IDs_range) = boundary

        # Create face field of same type provided by user (scalar or vector)
        sum = nothing
        if typeof(field) <: VectorField 
            faceField = FaceVectorField(mesh)
            sum = zeros(_get_float(mesh), 3) # create zero vector
        else
            faceField = FaceScalarField(mesh)
            sum = zero(_get_float(mesh)) # create zero
        end

        # Interpolate CFD results to boundary
        interpolate!(faceField, field, config)
        correct_boundaries!(faceField, field, field.BCs, time, config)

        # Calculate the average
        for fID ∈ IDs_range
            sum += faceField[fID]
        end
        ave = sum/length(IDs_range)

        # return average
        return ave
    end
"""
function boundary_average(patch::Symbol, field, fieldBCs, config; time=0)
    mesh = field.mesh

    ID = boundary_index(mesh.boundaries, patch)
    @info "calculating average on patch: $patch at index $ID"
    boundary = mesh.boundaries[ID]
    (; IDs_range) = boundary

    sum = nothing
    if typeof(field) <: VectorField 
        faceField = FaceVectorField(mesh)
        sum = zeros(_get_float(mesh), 3) # create zero vector
    else
        faceField = FaceScalarField(mesh)
        sum = zero(_get_float(mesh)) # create zero
    end
    interpolate!(faceField, field, config)
    correct_boundaries!(faceField, field, fieldBCs, time, config)

    for fID ∈ IDs_range
        sum += faceField[fID]
    end

    ave = sum/length(IDs_range)
    return ave
end

########### Must update
wall_shear_stress(patch::Symbol, model)  = begin
    # Line below needs to change to do selection based on nut BC
    turbulence = model.turbulence

    typeof(turbulence) <: RANS{Laminar} ? nut = ConstantScalar(0.0) : nut = model.turbulence.nut
    mesh = model.domain
    (; nu) = model.fluid
    (; U) = model.momentum
    (; boundaries, boundary_cellsID, faces) = mesh
    ID = boundary_index(boundaries, patch)
    boundary = boundaries[ID]
    (; IDs_range) = boundary
    @info "calculating viscous forces on patch: $patch at index $ID"
    x = FaceScalarField(zeros(Float64, length(IDs_range)), mesh)
    y = FaceScalarField(zeros(Float64, length(IDs_range)), mesh)
    z = FaceScalarField(zeros(Float64, length(IDs_range)), mesh)
    tauw = FaceVectorField(x,y,z, mesh)
    Uw = zero(_get_float(mesh))
    for i ∈ 1:length(boundaries.U)
        if ID == boundaries.U[i].ID
            Uw = boundaries.U[i].value
            surface_normal_gradient!(tauw, U, boundaries.U[i].value, IDs_range)
        end
    end

    pos = fill(SVector{3,Float64}(0,0,0), length(IDs_range))
    for i ∈ eachindex(tauw)
        # fID = facesID[i]
        # cID = cellsID[i]
        fID = IDs_range[i]
        cID = boundary_cellsID[fID]
        face = faces[fID]
        nueff = nu[cID]  + nut[cID]
        tauw.x[i] *= nueff # this may need using νtf? (wall funcs)
        tauw.y[i] *= nueff
        tauw.z[i] *= nueff
        pos[i] = face.centre
    end
    
    return tauw, pos
end

stress_tensor(U, ν, νt, config) = begin
    mesh = U.mesh
    TF = _get_float(mesh)
    gradU = Grad{Midpoint}(U)
    gradUT = T(gradU)
    Uf = FaceVectorField(U.mesh)
    grad!(gradU, Uf, U, boundaries.U, zero(TF), config) # assuming time=0
    # grad!(gradU, Uf, U, boundaries.U, , config)
    nueff = ScalarField(U.mesh) # temp variable
    nueff.values .= ν .+ νt.values
    Reff = TensorField(U.mesh)
    for i ∈ eachindex(Reff)
        Reff[i] = -nueff[i].*(gradU[i] .+ gradUT[i])
        # Reff[i] = -nueff[i].*(gradU[i])# .+ gradUT[i])
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
#     # surface_normal_gradient(snGrad, facesID, cellsID, U, boundaries.U[ID].value)
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