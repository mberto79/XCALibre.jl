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
    ndrange = length(cells)
    kernel! = _sized(div_kernel!, backend, workgroup, ndrange)
    kernel!(cells, F, cell_faces, cell_nsign, faces, phi, psif)
    # KernelAbstractions.synchronize(backend)

    # Retrieve number of boundary faces
    nbfaces = length(mesh.boundary_cellsID)

    # Launch boundary faces contribution kernel
    ndrange = nbfaces
    kernel! = _sized(div_boundary_faces_contribution_kernel!, backend, workgroup, ndrange)
    kernel!(faces, cells, phi, psif)
    # KernelAbstractions.synchronize(backend)
end

# Divergence calculation kernel

@kernel inbounds=true function div_kernel!(cells::AbstractArray{Cell{TF,SV,UR}}, F, cell_faces, cell_nsign, faces, phi, psif) where {TF,SV,UR}
    i = @index(Global)
    
    @inbounds begin
        # Extract required fields from cells structure
        volume, faces_range = cells.volume[i], cells.faces_range[i]
        
        # Set work item scalar field value as zero
        # phi.values[i] = 0.0 #zero(TF)
        reduction = zero(TF)
        # Loop over faces to iterate work item scalar field value 
        for fi ∈ faces_range
            # Extract face ID and corresponding normal direction
            fID = cell_faces[fi]
            nsign = cell_nsign[fi]

            # Extract required fields from faces structure
            area, normal = faces.area[fID], faces.normal[fID]

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
        cID = faces.ownerCells[i][1]
        volume = cells.volume[cID]
        area, normal = faces.area[i], faces.normal[i]

        # Boundary contribution calculation (boundary normals are correct by definition)
        Sf = area*normal
        Atomix.@atomic phi.values[cID] += psif[i]⋅Sf/volume
        # phi.values[cID] += psif[i]⋅Sf/volume
    end
end

# Divergence function definition - FaceScalarField

function div!(phi::ScalarField, psif::FaceScalarField, config)
    # Extract variables for function
    mesh = phi.mesh
    # backend = _get_backend(mesh)
    (; cells, cell_nsign, cell_faces, faces) = mesh
    (; hardware) = config
    (; backend, workgroup) = hardware

    # Retrieve user-selected float type
    F = _get_float(mesh)

    # Launch main calculation kernel
    ndrange = length(cells)
    kernel! = _sized(div_noS_kernel!, backend, workgroup, ndrange)
    kernel!(cells, F, cell_faces, cell_nsign, faces, phi, psif)
    # KernelAbstractions.synchronize(backend)

    # Retrieve number of boundary faces
    nbfaces = length(mesh.boundary_cellsID)

    # Launch boundary faces contribution kernel
    ndrange = nbfaces
    kernel! = _sized(div_noS_boundary_faces_contribution_kernel!, backend, workgroup, ndrange)
    kernel!(faces, cells, phi, psif)
    # KernelAbstractions.synchronize(backend)
end

# Divergence calculation kernel - FaceScalarField

@kernel function div_noS_kernel!(cells::AbstractArray{Cell{TF,SV,UR}}, F, cell_faces, cell_nsign, faces, phi, psif) where {TF,SV,UR}
    i = @index(Global)
    
    @inbounds begin
        # Extract required fields from cells structure
        volume, faces_range = cells.volume[i], cells.faces_range[i]
        
        # Set work item scalar field value as zero
        # phi.values[i] = 0.0 #zero(TF)
        reduction = zero(TF)
        # Loop over faces to iterate work item scalar field value 
        for fi ∈ faces_range
            # Extract face ID and corresponding normal direction
            fID = cell_faces[fi]
            nsign = cell_nsign[fi]

            # Extract required fields from faces structure
            area, normal = faces.area[fID], faces.normal[fID]

            # Atomix.@atomic phi.values[i] += psif[fID]⋅Sf*nsign/volume
            reduction += psif[fID]*nsign
        end
        phi.values[i] = reduction/volume # divide only once
    end
end

# Boundary faces contribution kernel

@kernel function div_noS_boundary_faces_contribution_kernel!(faces, cells, phi, psif)
    i = @index(Global)
    
    @inbounds begin
        # Retreive variables from work item boundary face
        cID = faces.ownerCells[i][1]
        volume = cells.volume[cID]
        area, normal = faces.area[i], faces.normal[i]

        # Boundary contribution calculation (boundary normals are correct by definition)
        # Sf = area*normal
        Atomix.@atomic phi.values[cID] += psif[i]/volume
        # phi.values[cID] += psif[i]⋅Sf/volume
    end
end
# Divergence of a cell tensor field scaled by a face coefficient

"""
    div!(phi::VectorField, Γf, tensor, BCs, config; Γc=nothing)

Cell values of ∇·(Γ tensor): `phi[i] = (1/V_i) Σ_f Γf[f] (tensor_f ⋅ S_f)`, where `tensor` is any
cell tensor field indexed as `tensor[i]` (including lazy forms such as `Dev2(T(gradU))`) and
`S_f` is the outward face area vector. At internal faces `tensor_f` is the linear interpolation
of the two cell values with the mesh weights; at boundary faces it is the owner-cell value.
With a cell coefficient `Γc` (any cell-indexable value, e.g. ν + ν_t) internal faces take the
interpolated product, `w Γc[P] tensor[P] + (1 - w) Γc[N] tensor[N]`, i.e. the face value of
the cell flux tensor Γ·tensor; boundary faces keep `Γf[f]` times the owner value. Interpolating
Γ and the tensor separately instead lets a large neighbour viscosity multiply the steep
gradient of a thin near-wall cell (a face weight close to 1 on the thin side), which makes an
explicit source built this way unstable.
Faces of `Slip`, `Symmetry` and `Empty` boundaries carry no flux. `BCs` are the boundary
conditions of the field `tensor` derives from (e.g. `boundaries.U`). Ghost cells of a
distributed mesh must hold current values of `tensor`.
"""
function div!(phi::VectorField, Γf, tensor, BCs, config; Γc=nothing)
    mesh = phi.mesh
    (; cells, cell_faces, cell_nsign, faces) = mesh
    (; backend, workgroup) = config.hardware

    phiv, Γfv = field_values(phi), _kernel_values(Γf)
    kernel! = _sized(_div_tensor_cells!, backend, workgroup, length(cells))
    kernel!(phiv, Γfv, Γc, tensor, cells, cell_faces, cell_nsign, faces)
    KernelAbstractions.synchronize(backend)

    for BC ∈ BCs
        _div_tensor_boundary!(phiv, Γfv, tensor, BC, cells, faces, backend, workgroup)
    end
    nothing
end

# kernel arguments are copied per thread: pass field storage, not the field and its mesh
_kernel_values(f::Union{ConstantScalar,ScalarField,FaceScalarField,VectorField,TensorField}) = field_values(f)
_kernel_values(f) = f

# each cell sums the fluxes through its internal faces
# Γ times the face tensor: face coefficient times interpolated tensor, or the interpolated
# product of cell coefficient and tensor
@inline _face_flux_tensor(Γf, ::Nothing, fID, wi, Γi, Ti, nID, Tn) = Γf[fID]*(wi*Ti + (one(wi) - wi)*Tn)
@inline _face_flux_tensor(Γf, Γc, fID, wi, Γi, Ti, nID, Tn) =
    wi*Γi*Ti + (one(wi) - wi)*_cell_coefficient(Γc, nID, typeof(wi))*Tn

@inline _cell_coefficient(::Nothing, i, ::Type{TF}) where TF = zero(TF)
@inline _cell_coefficient(Γc, i, ::Type{TF}) where TF = TF(Γc[i])

@kernel inbounds=true function _div_tensor_cells!(
    phi, Γf, Γc, tensor, cells::AbstractArray{Cell{TF,SV,UR}}, cell_faces, cell_nsign, faces
    ) where {TF,SV,UR}
    i = @index(Global)
    volume, faces_range = cells.volume[i], cells.faces_range[i]
    Ti = tensor[i]
    Γi = _cell_coefficient(Γc, i, TF)
    flux_sum = zero(SVector{3,TF})
    for fi ∈ faces_range
        fID = cell_faces[fi]
        nsign = cell_nsign[fi]
        ownerCells, area, normal, weight = faces.ownerCells[fID], faces.area[fID], faces.normal[fID], faces.weight[fID]
        # weight is the owner's (ownerCells[1]) share of the face value
        owner = ownerCells[1] == i
        wi = owner ? weight : one(TF) - weight
        nID = owner ? ownerCells[2] : ownerCells[1]
        ΓTf = _face_flux_tensor(Γf, Γc, fID, wi, Γi, Ti, nID, tensor[nID])
        flux_sum += (ΓTf*normal)*(area*nsign)
    end
    phi[i] = flux_sum/volume
end

_div_tensor_boundary!(phi, Γf, tensor, ::Union{Slip,Symmetry,Empty}, cells, faces, backend, workgroup) =
    nothing

# boundary faces add their flux to the owner cell (normals point out of the domain)
function _div_tensor_boundary!(phi, Γf, tensor, BC, cells, faces, backend, workgroup)
    (; IDs_range) = BC
    isempty(IDs_range) && return nothing
    kernel! = _sized(_div_tensor_boundary_kernel!, backend, workgroup, length(IDs_range))
    kernel!(phi, Γf, tensor, IDs_range, cells, faces)
    KernelAbstractions.synchronize(backend)
end

@kernel inbounds=true function _div_tensor_boundary_kernel!(phi, Γf, tensor, IDs_range, cells, faces)
    i = @index(Global)
    fID = IDs_range[i]
    cID = faces.ownerCells[fID][1]
    flux = Γf[fID]*(tensor[cID]*faces.normal[fID])*(faces.area[fID]/cells.volume[cID])
    Atomix.@atomic phi.x[cID] += flux[1]
    Atomix.@atomic phi.y[cID] += flux[2]
    Atomix.@atomic phi.z[cID] += flux[3]
end
