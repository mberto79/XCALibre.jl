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
    # Tf = eltype(grad.result.x.values)
    # SVector{3,Tf}(
    #     grad.result.x[i], 
    #     grad.result.y[i], 
    #     grad.result.z[i]
    #     )
    # Tf = eltype(grad.result.x.values)
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

# Vector field function definition

function grad!(grad::Grad{Orthogonal,F,R,I,M}, phif, phi, BCs, config) where {F,R<:VectorField,I,M}
    interpolate!(phif, phi, config)
    correct_boundaries!(phif, phi, BCs, config)
    green_gauss!(grad.result.x, grad.result.y, grad.result.z, phif, config)
end

# Tensor field function definition

function grad!(grad::Grad{Orthogonal,F,R,I,M}, psif, psi, BCs, config) where {F,R<:TensorField,I,M}
    interpolate!(psif, psi, config)
    correct_boundaries!(psif, psi, BCs, config)

    # Launch green-dauss for all tensor field dimensions
    green_gauss!(grad.result.xx, grad.result.yx, grad.result.zx, psif.x, config)
    green_gauss!(grad.result.xy, grad.result.yy, grad.result.zy, psif.y, config)
    green_gauss!(grad.result.xz, grad.result.yz, grad.result.zz, psif.z, config)
end

## Mid-point gradient calculation

# Scalar field calculation definition

function interpolate_midpoint!(phif::FaceScalarField, phi::ScalarField, config)
    # Extract required variables for function
    (; mesh) = phi
    (; faces) = mesh
    # backend = _get_backend(mesh)
    (; hardware) = config
    (; backend, workgroup) = hardware

    # Launch interpolate midpoint kernel for scalar field
    kernel! = interpolate_midpoint_scalar!(backend, workgroup)
    kernel!(faces, phif, phi, ndrange = length(faces))
    KernelAbstractions.synchronize(backend)
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

# Interpolate function for vector field definition (NEEDS KERNEL IMPLEMENTATION!!!!!!)

# Scalar field calculation definition

function interpolate_midpoint!(phif::FaceVectorField, phi::VectorField, config)
    # Extract required variables for function
    (; mesh) = phi
    (; faces) = mesh
    # backend = _get_backend(mesh)
    (; hardware) = config
    (; backend, workgroup) = hardware

    # Launch interpolate midpoint kernel for scalar field
    kernel! = interpolate_midpoint_vector!(backend, workgroup)
    kernel!(faces, phif, phi, ndrange = length(faces))
    KernelAbstractions.synchronize(backend)
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
        # weight = face.weight
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

# interpolate_midpoint!(psif::FaceVectorField, psi::VectorField, config) = begin
#     # Extract individual value vectors from vector field
#     (; x, y, z) = psif

#     # Retrieve mesh and faces
#     mesh = psi.mesh
#     faces = mesh.faces
    
#     # Set initial weight
#     weight = 0.5
    
#     @inbounds begin
#         # Loop over all faces to calculate interpolation
#         for fID ∈ eachindex(faces)
#             # Retrieve face, weight and ownerCells for loop iteration
#             face = faces[fID]
#             weight = face.weight
#             ownerCells = face.ownerCells
#             c1 = ownerCells[1]; c2 = ownerCells[2]
            
#             # Set values to interpolate between
#             x1 = psi.x[c1]; x2 = psi.x[c2]
#             y1 = psi.y[c1]; y2 = psi.y[c2]
#             z1 = psi.z[c1]; z2 = psi.z[c2]

#             # Interpolate calculation
#             x[fID] = weight*(x1 + x2)
#             y[fID] = weight*(y1 + y2)
#             z[fID] = weight*(z1 + z2)
#         end
#     end
# end

# Correct interpolation function definition

function correct_interpolation!(dx, dy, dz, phif, phi, config)
    # Define required variables for function
    (; mesh, values) = phif
    (; faces, cells) = mesh
    nbfaces = length(mesh.boundary_cellsID)
    phic = phi.values
    # backend = _get_backend(mesh)
    (; hardware) = config
    (; backend, workgroup) = hardware

    # Retrieve user-selected float type
    F = _get_float(mesh)
    
    # Set initial weight
    weight = 0.5

    # Launch correct interpolation kernel
    kernel! = correct_interpolation_kernel!(backend, workgroup)
    kernel!(faces, cells, nbfaces, phic, F, weight, dx, dy, dz, values, ndrange = length(faces)-nbfaces)
    KernelAbstractions.synchronize(backend)
end

# Correct interpolation kernel definition

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
    ∇phi1 = @inbounds SVector{3}(dx[owner1], dy[owner1], dz[owner1])
    ∇phi2 = @inbounds SVector{3}(dx[owner2], dy[owner2], dz[owner2])

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

# Vector field function definition

function grad!(grad::Grad{Midpoint,F,R,I,M}, phif, phi, BCs, config) where {F,R<:VectorField,I,M}
    interpolate_midpoint!(phif, phi, config)
    correct_boundaries!(phif, phi, BCs, config)
    green_gauss!(grad.result.x, grad.result.y, grad.result.z, phif, config)

    # Loop to run correction and green-gauss required number of times over all dimensions
    for i ∈ 1:2
        correct_interpolation!(grad.result.x, grad.result.y, grad.result.z, phif, phi, config)
        green_gauss!(grad.result.x, grad.result.y, grad.result.z, phif, config)
    end
end

# Tensor field function definition

function grad!(grad::Grad{Midpoint,F,R,I,M}, psif, psi, BCs, config) where {F,R<:TensorField,I,M}
    interpolate_midpoint!(psif, psi, config)
    correct_boundaries!(psif, psi, BCs, config)

    # Loop to run correction and green-gauss required number of times over all dimensions
    for i ∈ 1:2
    correct_interpolation!(grad.result.xx, grad.result.yx, grad.result.zx, psif.x, psi.x, config)
    green_gauss!(grad.result.xx, grad.result.yx, grad.result.zx, psif.x, config)
    
    correct_interpolation!(grad.result.xy, grad.result.yy, grad.result.zy, psif.y, psi.y, config)
    green_gauss!(grad.result.xy, grad.result.yy, grad.result.zy, psif.y, config)
    
    correct_interpolation!(grad.result.xz, grad.result.yz, grad.result.zz, psif.z, psi.z, config)
    green_gauss!(grad.result.xz, grad.result.yz, grad.result.zz, psif.z, config)
end
    # limit_gradient!(grad.result.xx, grad.result.yx, grad.result.zx, psi.x, config)
    # limit_gradient!(grad.result.xy, grad.result.yy, grad.result.zy, psi.y, config)
    # limit_gradient!(grad.result.xz, grad.result.yz, grad.result.zz, psi.z, config)
end


# TEMP LOCATION
function limit_gradient!(xv, yv, zv, F, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    mesh = F.mesh
    (; cells, cell_neighbours, cell_faces, cell_nsign, faces) = mesh

    minPhi0 = maximum(F.values) # use min value so all values compared are larger
    maxPhi0 = minimum(F.values)

    kernel! = _limit_gradient!(backend, workgroup)
    kernel!(xv, yv, zv, F, cells, cell_neighbours, cell_faces, cell_nsign, faces, minPhi0, maxPhi0, ndrange=length(cells))
    KernelAbstractions.synchronize(backend)
end

@kernel function _limit_gradient!(xv, yv, zv, F, cells, cell_neighbours, cell_faces, cell_nsign, faces, minPhi, maxPhi)
    cID = @index(Global)
    # mesh = F.mesh
    # (; cells, cell_neighbours, cell_faces, cell_nsign, faces) = mesh

    # minPhi0 = maximum(F.values) # use min value so all values compared are larger
    # maxPhi0 = minimum(F.values)

    # for (cID, cell) ∈ enumerate(cells)
        cell = cells[cID]
        # minPhi = minPhi0 # reset for next cell
        # maxPhi = maxPhi0

        # find min and max values around cell
        faces_range = cell.faces_range
        
        phiP = F[cID]
        # minPhi = phiP # reset for next cell
        # maxPhi = phiP
        for fi ∈ faces_range
            nID = cell_neighbours[fi]
            phiN = F[nID]
            maxPhi = max(phiN, maxPhi)
            minPhi = min(phiN, minPhi)
        end

        # g0 = ∇F[cID]
        x0 = xv[cID]
        y0 = yv[cID]
        z0 = zv[cID]
        g0 = SVector{3}(x0,y0,z0)
        cc = cell.centre

        for fi ∈ faces_range 
            fID = cell_faces[fi]
            face = faces[fID]
            nID = face.ownerCells[2]
            # phiN = F[nID]
            normal = face.normal
            nsign = cell_nsign[fi]
            na = nsign*normal

            fc = face.centre 
            cc_fc = fc - cc
            n0 = cc_fc/norm(cc_fc)
            gn = g0⋅n0
            δϕ = g0⋅cc_fc
            gτ = g0 - gn*n0
            if (maxPhi > phiP) && (δϕ > maxPhi - phiP)
                g0 = gτ + na*(maxPhi - phiP)
            elseif (minPhi < phiP) && (δϕ < minPhi - phiP)
                g0 = gτ + na*(minPhi - phiP)
            end            
        end
        # ∇F.result.x.values[cID] = g0[1]
        # ∇F.result.y.values[cID] = g0[2]
        # ∇F.result.z.values[cID] = g0[3]

        xv.values[cID] = g0[1]
        yv.values[cID] = g0[2]
        zv.values[cID] = g0[3]

        # x[cID] = g0[1]
        # y[cID] = g0[2]
        # z[cID] = g0[3]
    # end
end