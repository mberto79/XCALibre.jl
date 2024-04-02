export Grad
export grad!, source!
export get_scheme

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

# Grad outer contructor for scalar field

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

# Grad outer contructor for vector field

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
    Tf = eltype(grad.result.x.values)
    SVector{3,Tf}(
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
    Tf = eltype(tensor.xx.values)
    SMatrix{3,3,Tf,9}(
        tensor.xx[i],
        tensor.xy[i],
        tensor.xz[i],
        tensor.yx[i],
        tensor.yy[i],
        tensor.yz[i],
        tensor.zx[i],
        tensor.zy[i],
        tensor.zz[i],
        )
end

# GRADIENT CALCULATION FUNCTIONS

## Orthogonal (uncorrected) gradient calculation

# Vector field function

function grad!(grad::Grad{Orthogonal,F,R,I,M}, phif, phi, BCs) where {F,R<:VectorField,I,M}
    interpolate!(phif, phi)
    correct_boundaries!(phif, phi, BCs)
    green_gauss!(grad.result.x, grad.result.y, grad.result.z, phif)
end

# Tensor field function

function grad!(grad::Grad{Orthogonal,F,R,I,M}, psif, psi, BCs) where {F,R<:TensorField,I,M}
    interpolate!(psif, psi)
    correct_boundaries!(psif, psi, BCs)

    # Launch green-dauss for all tensor field dimensions
    green_gauss!(grad.result.xx, grad.result.yx, grad.result.zx, psif.x)
    green_gauss!(grad.result.xy, grad.result.yy, grad.result.zy, psif.y)
    green_gauss!(grad.result.xz, grad.result.yz, grad.result.zz, psif.z)
end

## Mid-point gradient calculation

# Scalar field calculation

function interpolate_midpoint!(phif::FaceScalarField, phi::ScalarField)
    # Extract required variables for function
    (; mesh) = phi
    (; faces) = mesh
    backend = _get_backend(mesh)

    # Launch interpolate midpoint kernel for scalar field
    kernel! = interpolate_midpoint_scalar!(backend)
    kernel!(faces, phif, phi, ndrange = length(faces))
    KernelAbstractions.synchronize(backend)
end

# Interpolate kernel for scalar field

@kernel function interpolate_midpoint_scalar!(faces, phif, phi)
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

# Interpolate function for vector field (NEEDS KERNEL IMPLEMENTATION!!!!!!)

interpolate_midpoint!(psif::FaceVectorField, psi::VectorField) = begin
    # Extract individual value vectors from vector field
    (; x, y, z) = psif

    # Retrieve mesh and faces
    mesh = psi.mesh
    faces = mesh.faces
    
    # Set initial weight
    weight = 0.5
    
    @inbounds begin
        # Loop over all faces to calculate interpolation
        for fID ∈ eachindex(faces)
            # Retrieve face, weight and ownerCells for loop iteration
            face = faces[fID]
            weight = face.weight
            ownerCells = face.ownerCells
            c1 = ownerCells[1]; c2 = ownerCells[2]
            
            # Set values to interpolate between
            x1 = psi.x[c1]; x2 = psi.x[c2]
            y1 = psi.y[c1]; y2 = psi.y[c2]
            z1 = psi.z[c1]; z2 = psi.z[c2]

            # Interpolate calculation
            x[fID] = weight*(x1 + x2)
            y[fID] = weight*(y1 + y2)
            z[fID] = weight*(z1 + z2)
        end
    end
end

# Correct interpolation function

function correct_interpolation!(dx, dy, dz, phif, phi)
    # Define required variables for function
    (; mesh, values) = phif
    (; faces, cells) = mesh
    nbfaces = length(mesh.boundary_cellsID)
    phic = phi.values
    backend = _get_backend(mesh)

    # Retrieve user-selected float type
    F = _get_float(mesh)
    
    # Set initial weight
    weight = 0.5

    # Launch correct interpolation kernel
    kernel! = correct_interpolation_kernel!(backend)
    kernel!(faces, cells, nbfaces, phic, F, weight, dx, dy, dz, values, ndrange = length(faces)-nbfaces)
    KernelAbstractions.synchronize(backend)
end

# Correct interpolation kernel

@kernel function correct_interpolation_kernel!(faces, cells, nbfaces, phic, F, weight, dx, dy, dz, values)
    i = @index(Global)
    # Set i such that it does not index boundary faces
    i += nbfaces

    # Retrieve fields from work item face
    (; ownerCells, centre) = faces[i]
    centre_faces = centre
    owner1 = ownerCells[1]
    owner2 = ownerCells[2]

    # Retrieve centre from work item cells and redefine variable name 
    (; centre) = cells[owner1]
    centre_cell1 = centre
    (; centre) = cells[owner2]
    centre_cell2 = centre

    # Retrieve values between which to correct interpolation
    phi1 = phic[owner1]
    phi2 = phic[owner2]

    # Allocate SVectors to use for gradient interpolation
    ∇phi1 = SVector{3, F}(dx[owner1], dy[owner1], dz[owner1])
    ∇phi2 = SVector{3, F}(dx[owner2], dy[owner2], dz[owner2])

    # Define variables as per interpolation mathematics convention
    rf = centre_faces
    rP = centre_cell1 
    rN = centre_cell2

    # Interpolate calculation
    phifᵖ = weight*(phi1 + phi2)
    ∇phi = weight*(∇phi1 + ∇phi2)
    Ri = rf - weight*(rP + rN)
    values[i] = phifᵖ + ∇phi⋅Ri
end

# Vector field function

function grad!(grad::Grad{Midpoint,F,R,I,M}, phif, phi, BCs) where {F,R<:VectorField,I,M}
    interpolate_midpoint!(phif, phi)
    correct_boundaries!(phif, phi, BCs)
    green_gauss!(grad.result.x, grad.result.y, grad.result.z, phif)

    # Loop to run correction and green-gauss required number of times over all dimensions
    for i ∈ 1:2
        correct_interpolation!(grad.result.x, grad.result.y, grad.result.z, phif, phi)
        green_gauss!(grad.result.x, grad.result.y, grad.result.z, phif)
    end
end

# Tensor field function

function grad!(grad::Grad{Midpoint,F,R,I,M}, psif, psi, BCs) where {F,R<:TensorField,I,M}
    interpolate_midpoint!(psif, psi)
    correct_boundaries!(psif, psi, BCs)

    # Loop to run correction and green-gauss required number of times over all dimensions
    for i ∈ 1:2
    correct_interpolation!(grad.result.xx, grad.result.yx, grad.result.zx, psif.x, psi.x)
    green_gauss!(grad.result.xx, grad.result.yx, grad.result.zx, psif.x)

    correct_interpolation!(grad.result.xy, grad.result.yy, grad.result.zy, psif.y, psi.y)
    green_gauss!(grad.result.xy, grad.result.yy, grad.result.zy, psif.y)

    correct_interpolation!(grad.result.xz, grad.result.yz, grad.result.zz, psif.z, psi.z)
    green_gauss!(grad.result.xz, grad.result.yz, grad.result.zz, psif.z)
    end
end