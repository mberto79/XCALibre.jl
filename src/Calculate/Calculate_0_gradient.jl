export Grad
export grad!, source!
export get_scheme
export grad_old!

# Define Gradient type and functionality

struct Grad{S<:AbstractScheme,F,R,I,M} <: AbstractField
    field::F
    result::R
    correctors::I
    correct::Bool
    mesh::M
end
function Adapt.adapt_structure(to, itp::Grad{S}) where {S}
    field = Adapt.adapt_structure(to, itp.field); F = typeof(field)
    result = Adapt.adapt_structure(to, itp.result); R = typeof(result)
    correctors = Adapt.adapt_structure(to, itp.correctors); I = typeof(correctors)
    correct = Adapt.adapt_structure(to, itp.correct)
    mesh = Adapt.adapt_structure(to, itp.mesh); M = typeof(mesh)
    Grad{S,F,R,I,M}(field, result, correctors, correct, mesh)
end

# Grad outer constructor for scalar field definition
Grad{S}(phi::ScalarField) where S= begin
    # Retrieve mesh and define grad as vector field
    mesh = phi.mesh
    grad = VectorField(mesh)

    # Retrieve user-selected types
    F = typeof(phi)
    R = typeof(grad)
    I = _get_int(mesh)
    M = typeof(mesh)

    # Construct Grad
    Grad{S,F,R,I,M}(phi, grad, one(I), false, mesh)
end

# Grad outer constructor for vector field definition
Grad{S}(psi::VectorField) where S = begin
    # Retrieve mesh and define grad as tensor field
    mesh = psi.mesh
    tgrad = TensorField(mesh)

    # Retrieve user-selected types
    F = typeof(psi)
    R = typeof(tgrad)
    I = _get_int(mesh)
    M = typeof(mesh)

    # Construct Grad
    Grad{S,F,R,I,M}(psi, tgrad, one(I), false, mesh)
end

Base.getindex(grad::Grad{S,F,R,I,M}, i::Integer) where {S,F,R<:VectorField,I,M} = begin
    @inbounds SVector{3}(
        grad.result.x[i], 
        grad.result.y[i], 
        grad.result.z[i]
        )
end

Base.getindex(grad::Grad{S,F,R,I,M}, i::Integer) where {S,F,R<:AbstractTensorField,I,M} = begin
    Tf = eltype(grad.result.xx.values)
    tensor = grad.result
    SMatrix{3,3,Tf,9}(
        tensor.xx[i],
        tensor.yx[i],
        tensor.zx[i],
        tensor.xy[i],
        tensor.yy[i],
        tensor.zy[i],
        tensor.xz[i],
        tensor.yz[i],
        tensor.zz[i],
        )
end

Base.getindex(t::T{Grad{S,F,R,I,M}}, i::Integer) where {S,F,R<:AbstractTensorField,I,M} = begin
    tensor = t.parent.result
    return tensor[i]'
end

# GRADIENT CALCULATION FUNCTIONS

## Orthogonal (uncorrected) gradient calculation

function grad!(grad::Grad{Orthogonal,F,R,I,M}, phif, phi, BCs, time, config) where {F,R<:VectorField,I,M}
    interpolate!(phif, phi, config)
    correct_boundaries!(phif, phi, BCs, time, config)
    green_gauss!(grad, phif, config)
end

# Tensor field function definition

function grad!(grad::Grad{Orthogonal,F,R,I,M}, psif, psi, BCs, time, config) where {F,R<:TensorField,I,M}
    interpolate!(psif, psi, config)
    correct_boundaries!(psif, psi, BCs, time, config)
    green_gauss!(grad, psif, config)
end

## Mid-point gradient calculation

# Scalar field calculation definition

function interpolate_midpoint!(phif::FaceScalarField, phi::ScalarField, config)
    # Extract required variables for function
    (; mesh) = phi
    (; faces) = mesh
    (; hardware) = config
    (; backend, workgroup) = hardware

    # Launch interpolate midpoint kernel for scalar field
    kernel! = interpolate_midpoint_scalar!(backend, workgroup)
    kernel!(faces, phif, phi, ndrange = length(faces))
    # # KernelAbstractions.synchronize(backend)
end

# Interpolate kernel for scalar field definition

@kernel function interpolate_midpoint_scalar!(
    faces, phif::FaceScalarField, phi::ScalarField)
    i = @index(Global)

    @inbounds begin
        # Extract required fields from work item face and define ownerCell variables
        (; ownerCells) = faces[i]
        c1 = ownerCells[1]
        c2 = ownerCells[2]

        # Interpolate calculation
        phif[i] = 0.5*(phi[c1] + phi[c2])
    end
end

# Scalar field calculation definition

function interpolate_midpoint!(phif::FaceVectorField, phi::VectorField, config)
    # Extract required variables for function
    (; mesh) = phi
    (; faces) = mesh
    (; hardware) = config
    (; backend, workgroup) = hardware

    # Launch interpolate midpoint kernel for scalar field
    kernel! = interpolate_midpoint_vector!(backend, workgroup)
    kernel!(faces, phif, phi, ndrange = length(faces))
    # # KernelAbstractions.synchronize(backend)
end

@kernel function interpolate_midpoint_vector!(
    faces, psif::FaceVectorField, psi::VectorField)
    fID = @index(Global)

    @uniform begin
        # Extract individual value vectors from vector field
        (; x, y, z) = psif
        weight = 0.5
    end

    @inbounds begin
        # Retrieve face, weight and ownerCells for loop iteration
        face = faces[fID]
        ownerCells = face.ownerCells
        c1 = ownerCells[1]; c2 = ownerCells[2]
        
        # Set values to interpolate between
        psi1 = psi[c1]
        psi2 = psi[c2]
        midpoint = weight*(psi1 + psi2)

        # Interpolate calculation
        x[fID] = midpoint[1]
        y[fID] = midpoint[2]
        z[fID] = midpoint[3]
    end
end

# Correct interpolation function definition

function correct_interpolation!(grad, phif, phi, config)
    # Define required variables for function
    (; mesh) = phif
    (; faces, cells) = mesh
    nbfaces = length(mesh.boundary_cellsID)
    (; hardware) = config
    (; backend, workgroup) = hardware

    # Retrieve user-selected float type
    F = _get_float(mesh)
    
    # Set initial weight
    weight = 0.5

    # Launch correct interpolation kernel
    kernel! = correct_interpolation_kernel!(backend, workgroup)
    kernel!(faces, cells, nbfaces, phi, F, weight, grad, phif, ndrange = length(faces)-nbfaces)
    # # KernelAbstractions.synchronize(backend)
end

# Correct interpolation kernel definition

@kernel function correct_interpolation_kernel!(faces, cells, nbfaces, phi, F, weight, grad, phif::Field) where {Field}
    i = @index(Global)
    i += nbfaces # Set i such that it does not index boundary faces

    # Retrieve fields from work item face
    (; ownerCells, centre) = faces[i]
    centre_face = centre
    owner1 = ownerCells[1]
    owner2 = ownerCells[2]

    # Retrieve centre from work item cells and redefine variable name 
    (; centre) = cells[owner1]
    centre_cell1 = centre
    (; centre) = cells[owner2]
    centre_cell2 = centre

    # Retrieve values between which to correct interpolation
    phi1 = phi[owner1]
    phi2 = phi[owner2]

    ∇phi1 = grad[owner1]
    ∇phi2 = grad[owner2]

    # Define variables as per interpolation mathematics convention
    rf = centre_face
    rP = centre_cell1 
    rN = centre_cell2

    # Interpolate calculation
    phifᵖ = weight*(phi1 + phi2)
    ∇phi = weight*(∇phi1 + ∇phi2)
    Ri = rf - weight*(rP + rN)

    if Field <: AbstractScalarField
        phif[i] = phifᵖ + ∇phi⋅Ri   # vector/vector => returns scalar
    else
        phif[i] = phifᵖ + ∇phi*Ri  # tensor/vector => returns vector
    end
      
end

function grad!(grad::Grad{Midpoint,F,R,I,M}, phif, phi, BCs, time, config) where {F,R,I,M}
    interpolate_midpoint!(phif, phi, config)
    correct_boundaries!(phif, phi, BCs, time, config)
    green_gauss!(grad, phif, config)

    # Loop to run correction and green-gauss required number of times over all dimensions
    for i ∈ 1:2
        correct_interpolation!(grad, phif, phi, config)
        green_gauss!(grad, phif, config)
    end
end