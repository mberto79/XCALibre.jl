export Div
export div! 

# Define Divergence type and functionality

struct Div{VF<:VectorField,FVF<:FaceVectorField,F,M}
    vector::VF
    face_vector::FVF
    values::Vector{F}
    mesh::M
end
Adapt.@adapt_structure Div
Div(vector::VectorField) = begin
    mesh = vector.mesh
    face_vector = FaceVectorField(mesh)
    values = zeros(F, length(mesh.cells))
    Div(vector, face_vector, values, mesh)
end

# Divergence function definition

function div!(phi::ScalarField, psif::FaceVectorField, config)
    # Extract variables for function
    mesh = phi.mesh
    # backend = _get_backend(mesh)
    (; cells, cell_nsign, cell_faces, faces) = mesh
    (; hardware) = config
    (; backend, workgroup) = hardware

    # Retrieve user-selected float type
    F = _get_float(mesh)

    # Launch main calculation kernel
    kernel! = div_kernel!(backend, workgroup)
    kernel!(cells, F, cell_faces, cell_nsign, faces, phi, psif, ndrange = length(cells))
    KernelAbstractions.synchronize(backend)

    # Retrieve number of boundary faces
    nbfaces = length(mesh.boundary_cellsID)

    # Launch boundary faces contribution kernel
    kernel! = div_boundary_faces_contribution_kernel!(backend, workgroup)
    kernel!(faces, cells, phi, psif, ndrange = nbfaces)
    KernelAbstractions.synchronize(backend)
end

# Divergence calculation kernel

@kernel function div_kernel!(cells::AbstractArray{Cell{TF,SV,UR}}, F, cell_faces, cell_nsign, faces, phi, psif) where {TF,SV,UR}
    i = @index(Global)
    
    @inbounds begin
        # Extract required fields from cells structure
        (; volume, faces_range) = cells[i]
        
        # Set work item scalar field value as zero
        # phi.values[i] = 0.0 #zero(TF)
        reduction = zero(TF)
        # Loop over faces to iterate work item scalar field value 
        for fi ∈ faces_range
            # Extract face ID and corresponding normal direction
            fID = cell_faces[fi]
            nsign = cell_nsign[fi]

            # Extract required fields from faces structure
            (; area, normal) = faces[fID]

            # Scalar field values calculation
            Sf = area*normal
            # Atomix.@atomic phi.values[i] += psif[fID]⋅Sf*nsign/volume
            reduction += psif[fID]⋅Sf*nsign
        end
        phi.values[i] = reduction/volume # divide only once
    end
end

# Boundary faces contribution kernel

@kernel function div_boundary_faces_contribution_kernel!(faces, cells, phi, psif)
    i = @index(Global)
    
    @inbounds begin
        # Retreive variables from work item boundary face
        cID = faces[i].ownerCells[1]
        volume = cells[cID].volume
        (; area, normal) = faces[i]

        # Boundary contribution calculation (boundary normals are correct by definition)
        Sf = area*normal
        Atomix.@atomic phi.values[cID] += psif[i]⋅Sf/volume
        # phi.values[cID] += psif[i]⋅Sf/volume
    end
end