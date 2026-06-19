export boundary_average
export pressure_force, viscous_force
export stress_tensor, wall_shear_stress

"""
    pressure_force(patch::Symbol, model, config)

Calculate the pressure force vector `[Fx, Fy, Fz]` acting on a given patch/boundary.

# Input arguments

- `patch::Symbol` name of the boundary of interest (as a `Symbol`)
- `model` `Physics` object defining the simulation
- `config` `Configuration` object (provides the hardware backend)
"""
function pressure_force(patch::Symbol, model, config)
    mesh = model.domain
    p = model.momentum.p
    TF = _get_float(mesh)
    # incompressible p is kinematic (p/ρ) so scale by reference density; compressible p is absolute
    rhoref = model.fluid isa AbstractIncompressible ? model.fluid.rho : ConstantScalar(one(TF))
    (; backend, workgroup) = config.hardware

    ID = boundary_index(model.boundary_info, patch)
    @info "calculating pressure force on patch: $patch at index $ID"
    IDs_range = get_boundaries(mesh.boundaries)[ID].IDs_range
    (; faces, boundary_cellsID) = mesh

    n = length(IDs_range)
    fx = KernelAbstractions.zeros(backend, TF, n)
    fy = KernelAbstractions.zeros(backend, TF, n)
    fz = KernelAbstractions.zeros(backend, TF, n)

    kernel! = _pressure_force!(_setup(backend, workgroup, n)...)
    kernel!(fx, fy, fz, p, rhoref, faces, boundary_cellsID, IDs_range)
    KernelAbstractions.synchronize(backend)

    Fp = TF[sum(fx), sum(fy), sum(fz)]
    @info "Pressure force on $patch: ($(Fp[1]), $(Fp[2]), $(Fp[3]))"
    return Fp
end

@kernel function _pressure_force!(fx, fy, fz, p, rhoref, faces, boundary_cellsID, IDs_range)
    i = @index(Global)
    @inbounds begin
        fID = IDs_range[i]
        cID = boundary_cellsID[fID]
        (; area, normal) = faces[fID]
        flux = rhoref[cID]*p[cID]*(area*normal)
        fx[i] = flux[1]
        fy[i] = flux[2]
        fz[i] = flux[3]
    end
end

"""
    viscous_force(patch::Symbol, model, config)

Calculate the viscous force vector `[Fx, Fy, Fz]` acting on a given patch/boundary.

# Input arguments

- `patch::Symbol` name of the boundary of interest (as a `Symbol`)
- `model` `Physics` object defining the simulation
- `config` `Configuration` object (provides boundary conditions and hardware backend)
"""
function viscous_force(patch::Symbol, model, config)
    mesh = model.domain
    U = model.momentum.U
    nu = model.fluid.nu
    rho = model.fluid.rho
    TF = _get_float(mesh)
    nut = model.turbulence isa Laminar ? ConstantScalar(zero(TF)) : model.turbulence.nutf # wall-face nut (wall funcs)
    (; backend, workgroup) = config.hardware

    ID = boundary_index(model.boundary_info, patch)
    @info "calculating viscous force on patch: $patch at index $ID"
    Uw = _patch_bc_value(config.boundaries.U, ID) # wall velocity from the U boundary condition
    IDs_range = get_boundaries(mesh.boundaries)[ID].IDs_range
    (; faces, boundary_cellsID) = mesh

    n = length(IDs_range)
    fx = KernelAbstractions.zeros(backend, TF, n)
    fy = KernelAbstractions.zeros(backend, TF, n)
    fz = KernelAbstractions.zeros(backend, TF, n)

    kernel! = _viscous_force!(_setup(backend, workgroup, n)...)
    kernel!(fx, fy, fz, U, Uw, nu, nut, rho, faces, boundary_cellsID, IDs_range)
    KernelAbstractions.synchronize(backend)

    Fv = TF[sum(fx), sum(fy), sum(fz)]
    @info "Viscous force on $patch: ($(Fv[1]), $(Fv[2]), $(Fv[3]))"
    return Fv
end

@kernel function _viscous_force!(
    fx, fy, fz, U, Uw, nu, nut, rho, faces, boundary_cellsID, IDs_range)
    i = @index(Global)
    @inbounds begin
        fID = IDs_range[i]
        cID = boundary_cellsID[fID]
        (; area, normal, delta, e) = faces[fID]
        Udiff = U[cID] - Uw
        Up = Udiff - (Udiff⋅normal)*normal # parallel velocity difference
        dperp = delta*(e⋅normal) # wall-normal distance (projection of cell-to-face vector)
        snGrad = Up/dperp
        coeff = rho[cID]*area*(nu[cID] + nut[fID]) # nut is wall-face value νtf (wall funcs)
        fx[i] = snGrad[1]*coeff
        fy[i] = snGrad[2]*coeff
        fz[i] = snGrad[3]*coeff
    end
end

# Host-side lookup of the BC value matching a boundary ID (BCs is a heterogeneous tuple)
_patch_bc_value(BCs, ID) = begin
    for bc ∈ BCs
        bc.ID == ID && return bc.value
    end
    error("No boundary condition found for boundary index $ID")
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

"""
    wall_shear_stress(patch::Symbol, model,config)

Function to calculate the wall shear stress acting on a given patch/boundary.

# Input arguments

- `patch::Symbol` name of the boundary of interest (as a `Symbol`)
- `model` instance of `Physics` object needs to be passed 
- `config` need to pass `Configuration` object as this contains the boundary conditions
"""
wall_shear_stress(patch::Symbol, model,config)  = begin
    # Line below needs to change to do selection based on nut BC
    turbulence = model.turbulence
    UBCs = config.boundaries.U
    typeof(turbulence) <: Laminar ? nut = ConstantScalar(0.0) : nut = model.turbulence.nutf # wall-face νtf
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
    for i ∈ 1:length(UBCs)
        if ID == UBCs[i].ID
            Uw = UBCs[i].value
            surface_normal_gradient!(tauw, U, UBCs[i].value, IDs_range)
        end
    end

    pos = fill(SVector{3,Float64}(0,0,0), length(IDs_range))
    for i ∈ eachindex(tauw)
        # fID = facesID[i]
        # cID = cellsID[i]
        fID = IDs_range[i]
        cID = boundary_cellsID[fID]
        face = faces[fID]
        nueff = nu[cID]  + nut[fID] # nut is wall-face value νtf (wall funcs)
        tauw.y[i] *= nueff
        tauw.z[i] *= nueff
        pos[i] = face.centre
    end
    
    return tauw, pos
end
"""
    stress_tensor(U::VectorField, ν, νt, config)

Function to calculate the stress tensor.

# Input arguments

- `U::VectorField` velocity field
- `ν` laminar viscosity of the fluid
- `νt` eddy viscosity from turbulence models. Pass ConstantScalar(0) for laminar flows
- `config` need to pass `Configuration` object as this contains the boundary conditions
"""
stress_tensor(U, ν, νt, config) = begin
    mesh = U.mesh
    TF = _get_float(mesh)
    gradU = Grad{Midpoint}(U)
    gradUT = T(gradU)
    Uf = FaceVectorField(U.mesh)
    grad!(gradU, Uf, U, config.boundaries.U, zero(TF), config) # assuming time=0
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