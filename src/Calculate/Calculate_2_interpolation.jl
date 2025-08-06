# export correct_boundaries!
export interpolate!
export interpolate_harmonic!

# Temporary functions to extract boundary array
function to_cpu(boundaries::AbstractArray)
    return boundaries
end

# Function to copy from GPU to CPU
function to_cpu(boundaries::AbstractGPUArray)
    # Copy boundaries to CPU
    boundaries_cpu = Array{eltype(boundaries)}(undef, length(boundaries))
    KernelAbstractions.copyto!(CPU(), boundaries_cpu, boundaries)
    return boundaries_cpu
end

# Function to correct interpolation at boundaries (expands loop to reduce allocations)


# @generated function correct_boundaries!(phif, phi, BCs, time, config)
#     unpacked_BCs = []
#     for i ∈ 1:length(BCs.parameters)
#         unpack = quote
#             #KERNEL LAUNCH
#             adjust_boundary!(BCs[$i], phif, phi, boundaries, boundary_cellsID, time, backend, workgroup)
#         end
#         push!(unpacked_BCs, unpack)
#     end
#     quote
#     (; mesh) = phif
#     (; boundary_cellsID, boundaries) = mesh 
#     (; hardware) = config
#     (; backend, workgroup) = hardware
#     $(unpacked_BCs...) 
#     end
# end

## SCALAR INTERPOLATION

function interpolate!(phif::FaceScalarField, phi::ScalarField, config)
    # Extract values arrays from scalar fields 
    vals = phi.values
    fvals = phif.values

    # Extract faces from mesh
    mesh = phif.mesh
    (; cells, faces) = mesh

    # Launch interpolate kernel
    (; hardware) = config
    (; backend, workgroup) = hardware
    ndrange = length(faces)
    kernel! = interpolate_Scalar!(_setup(backend, workgroup, ndrange)...)
    kernel!(fvals, vals, cells, faces)
    # # KernelAbstractions.synchronize(backend)
end

@kernel function interpolate_Scalar!(fvals, vals, cells, faces)
    # Define index for thread
    i = @index(Global)

    @inbounds begin
        # Deconstruct faces to use weight and ownerCells in calculations
        face = faces[i]
        (; weight, ownerCells, normal) = face
        F = face.centre

        # Calculate initial values based on index queried from ownerCells
        owner1 = ownerCells[1]
        owner2 = ownerCells[2]
        phi1 = vals[owner1]
        phi2 = vals[owner2]

        one_minus_weight = 1.0 - weight
        fvals[i] = weight*phi1 + one_minus_weight*phi2 # check weight is used correctly!
    end
end



## HARMONIC SCALAR INTERPOLATION

function interpolate_harmonic!(phif::FaceScalarField, phi::ScalarField, config)
    # Extract values arrays from scalar fields 
    vals = phi.values
    fvals = phif.values

    # Extract faces from mesh
    mesh = phif.mesh
    (; cells, faces) = mesh

    # Launch interpolate kernel
    (; hardware) = config
    (; backend, workgroup) = hardware
    ndrange = length(faces)
    kernel! = interpolate_harmonic_Scalar!(_setup(backend, workgroup, ndrange)...)
    kernel!(fvals, vals, cells, faces)
    # # KernelAbstractions.synchronize(backend)
end

@kernel function interpolate_harmonic_Scalar!(fvals, vals, cells, faces)
    # Define index for thread
    i = @index(Global)

    @inbounds begin
        # Deconstruct faces to use weight and ownerCells in calculations
        face = faces[i]
        (; weight, ownerCells, normal) = face
        F = face.centre

        # Calculate initial values based on index queried from ownerCells
        owner1 = ownerCells[1]
        owner2 = ownerCells[2]
        phi1 = vals[owner1]
        phi2 = vals[owner2]

        # one_minus_weight = 1.0 - weight
        
        fvals[i] = 2*((phi1*phi2)/(phi1+phi2))
    end
end




# VECTOR INTERPOLATION
function interpolate!(psif::FaceVectorField, psi::VectorField, config)
    # Extract x, y, z, values from FaceVectorField
    (; mesh) = psif

    #Redefine x, y, z values to be used in kernel
    xf = psif.x
    yf = psif.y
    zf = psif.z

    # Extract x, y, z, values from VectorField
    xv = psi.x
    yv = psi.y
    zv = psi.z

    #Extract faces array from mesh
    faces = mesh.faces

    # Launch interpolate kernel
    # backend = _get_backend(mesh)
    (; hardware) = config
    (; backend, workgroup) = hardware
    ndrange = length(faces)
    kernel! = interpolate_Vector!(_setup(backend, workgroup, ndrange)...)
    kernel!(xv, yv, zv, xf, yf, zf, faces)
    # # KernelAbstractions.synchronize(backend)
end

@kernel function interpolate_Vector!(@Const(xv), @Const(yv), @Const(zv), xf, yf, zf, @Const(faces))
    # Define index for thread
    i = @index(Global)

    @inbounds begin
        # Deconstruct faces to use weight and ownerCells in calculations
        # @synchronize # commented out on 2024/10/24
        (; weight, ownerCells) = faces[i]

        # Define indices for initial x and y values from psi struct
        cID1 = ownerCells[1]; cID2 = ownerCells[2]
        x1 = xv[cID1]; x2 = xv[cID2]
        y1 = yv[cID1]; y2 = yv[cID2]
        z1 = zv[cID1]; z2 = zv[cID2]

        # Calculate one minus weight
        one_minus_weight = 1.0 - weight

        # Update psif x and y arrays for interpolation (IMPLEMENT 3D)
        xf[i] = weight*x1 + one_minus_weight*x2 # check weight is used correctly!
        yf[i] = weight*y1 + one_minus_weight*y2 # check weight is used correctly!
        zf[i] = weight*z1 + one_minus_weight*z2 # check weight is used correctly!
    end
end

# GRADIENT INTERPOLATION

function interpolate!(
    gradf::FaceVectorField, grad::Grad, phi
    )
    (; mesh, x, y, z) = gradf
    (; cells, faces) = mesh
    (; values) = phi
    nbfaces = total_boundary_faces(mesh)
    start = nbfaces + 1
    @inbounds for fID ∈ start:length(faces)
        face = faces[fID]
        (; delta, ownerCells, e) = face
        cID1 = ownerCells[1]
        cID2 = ownerCells[2]
        grad1 = grad(cID1)
        grad2 = grad(cID2)
        # get weight for current scheme
        w, df = weight(get_scheme(grad), cells, faces, fID)
        one_minus_weight = 1.0 - w
        # calculate interpolated value
        grad_ave = w*grad1 + one_minus_weight*grad2
        # correct interpolation
        grad_corr = grad_ave + ((values[cID2] - values[cID1])/delta - (grad_ave⋅e))*e
        x[fID] = grad_corr[1]
        y[fID] = grad_corr[2]
        z[fID] = grad_corr[3]
    end
end

