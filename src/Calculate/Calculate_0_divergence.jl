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

    # Launch boundary faces contribution kernel: one work item per boundary cell
    (; boundary_cells, boundary_cell_faces_range, boundary_cell_faces) = mesh
    ndrange = length(boundary_cells)
    kernel! = _sized(div_boundary_faces_contribution_kernel!, backend, workgroup, ndrange)
    kernel!(faces, cells, phi, psif, boundary_cells, boundary_cell_faces_range, boundary_cell_faces)
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

# each boundary cell adds its boundary faces in face order (no atomics: deterministic)
@kernel function div_boundary_faces_contribution_kernel!(
    faces, cells, phi, psif, boundary_cells, boundary_cell_faces_range, boundary_cell_faces)
    j = @index(Global)
    
    @inbounds begin
        cID = boundary_cells[j]
        volume = cells.volume[cID]
        for k ∈ boundary_cell_faces_range[j]
            i = boundary_cell_faces[k]
            area, normal = faces.area[i], faces.normal[i]

            # Boundary contribution calculation (boundary normals are correct by definition)
            Sf = area*normal
            phi.values[cID] += psif[i]⋅Sf/volume
        end
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

    # Launch boundary faces contribution kernel: one work item per boundary cell
    (; boundary_cells, boundary_cell_faces_range, boundary_cell_faces) = mesh
    ndrange = length(boundary_cells)
    kernel! = _sized(div_noS_boundary_faces_contribution_kernel!, backend, workgroup, ndrange)
    kernel!(faces, cells, phi, psif, boundary_cells, boundary_cell_faces_range, boundary_cell_faces)
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

# each boundary cell adds its boundary faces in face order (no atomics: deterministic)
@kernel function div_noS_boundary_faces_contribution_kernel!(
    faces, cells, phi, psif, boundary_cells, boundary_cell_faces_range, boundary_cell_faces)
    j = @index(Global)
    
    @inbounds begin
        cID = boundary_cells[j]
        volume = cells.volume[cID]
        for k ∈ boundary_cell_faces_range[j]
            phi.values[cID] += psif[boundary_cell_faces[k]]/volume
        end
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
the cell flux tensor Γ·tensor; boundary faces keep `Γf[f]` times the owner value, and periodic
faces take the internal-face value between the two cells of the pair. Interpolating
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
        _div_tensor_boundary!(phiv, Γfv, Γc, tensor, BC, mesh, backend, workgroup)
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

_div_tensor_boundary!(phi, Γf, Γc, tensor, ::Union{Slip,Symmetry,Empty}, mesh, backend, workgroup) =
    nothing

# periodic faces take the internal-face value from the two cells of the pair, so the fluxes of
# matched faces are equal and opposite
_div_tensor_boundary!(phi, Γf, Γc, tensor, BC::Union{Periodic,PeriodicParent}, mesh, backend, workgroup) =
    div_tensor_periodic!(phi, Γf, Γc, tensor, BC, mesh, backend, workgroup)

# boundary faces add their flux to the owner cell (normals point out of the domain); one work item
# per boundary cell adds its faces of this patch in face order, so the sum is deterministic
function _div_tensor_boundary!(phi, Γf, Γc, tensor, BC, mesh, backend, workgroup)
    (; IDs_range) = BC
    isempty(IDs_range) && return nothing
    (; cells, faces, boundary_cells, boundary_cell_faces_range, boundary_cell_faces) = mesh
    kernel! = _sized(_div_tensor_boundary_kernel!, backend, workgroup, length(boundary_cells))
    kernel!(phi, Γf, tensor, IDs_range, cells, faces,
        boundary_cells, boundary_cell_faces_range, boundary_cell_faces)
    KernelAbstractions.synchronize(backend)
end

@kernel inbounds=true function _div_tensor_boundary_kernel!(phi, Γf, tensor, IDs_range, cells, faces,
    boundary_cells, boundary_cell_faces_range, boundary_cell_faces)
    j = @index(Global)
    cID = boundary_cells[j]
    for k ∈ boundary_cell_faces_range[j]
        fID = boundary_cell_faces[k]
        fID ∈ IDs_range || continue
        flux = Γf[fID]*(tensor[cID]*faces.normal[fID])*(faces.area[fID]/cells.volume[cID])
        phi.x[cID] += flux[1]
        phi.y[cID] += flux[2]
        phi.z[cID] += flux[3]
    end
end

# Periodic face: Γ (parent face) times the tensor interpolated between the owner cells of the pair
# with the weights of `correct_interpolation_periodic`; both faces get the same face tensor, so
# their fluxes cancel as at an internal face.
function div_tensor_periodic!(phi, Γf, Γc, tensor, BC, mesh, backend, workgroup)
    (; IDs_range) = BC
    isempty(IDs_range) && return nothing
    (; cells, faces, boundary_cells, boundary_cell_faces_range, boundary_cell_faces) = mesh
    kernel! = _sized(_div_tensor_periodic_kernel!, backend, workgroup, length(boundary_cells))
    kernel!(phi, Γf, Γc, tensor, IDs_range, BC.value.face_map, BC isa PeriodicParent, cells, faces,
        boundary_cells, boundary_cell_faces_range, boundary_cell_faces)
    KernelAbstractions.synchronize(backend)
end

# one work item per boundary cell adds its faces of this patch in face order (deterministic)
@kernel inbounds=true function _div_tensor_periodic_kernel!(
    phi, Γf, Γc, tensor, IDs_range, face_map, isparent, cells::AbstractArray{Cell{TF,SV,UR}}, faces,
    boundary_cells, boundary_cell_faces_range, boundary_cell_faces) where {TF,SV,UR}
    j = @index(Global)
    cID = boundary_cells[j]
    for k ∈ boundary_cell_faces_range[j]
        fID = boundary_cell_faces[k]
        fID ∈ IDs_range || continue
        pfID = face_map[fID - first(IDs_range) + 1]
        pcID = faces.ownerCells[pfID][1]
        delta, pdelta = faces.delta[fID], faces.delta[pfID]
        w = pdelta/(delta + pdelta) # share of the owner cell
        ΓTf = _face_flux_tensor(Γf, Γc, isparent ? fID : pfID, w, _cell_coefficient(Γc, cID, TF), tensor[cID], pcID, tensor[pcID])
        flux = (ΓTf*faces.normal[fID])*(faces.area[fID]/cells.volume[cID])
        phi.x[cID] += flux[1]
        phi.y[cID] += flux[2]
        phi.z[cID] += flux[3]
    end
end

# Boundary-face velocity gradient shared by the momentum transpose stress and the enthalpy viscous
# work: face-normal gradient of U at face fID and `time`, or nothing to keep the owner-cell one.
# Fixed-value conditions: (U_b - U_c)/delta with the boundary value U_b of the face
@inline _boundary_sngrad(BC::Union{Wall,Dirichlet}, Uc, delta, normal, faces, fID, time) =
    (SVector{3}(BC.value[1], BC.value[2], BC.value[3]) - Uc)/delta
# user function f(coords, time, index), evaluated as by the boundary assembly and interpolation
@inline function _boundary_sngrad(BC::DirichletFunction, Uc, delta, normal, faces, fID, time)
    i = fID - BC.IDs_range.start + 1
    (convert(typeof(Uc), BC.value(faces.centre[fID], time, i)) - Uc)/delta
end
@inline _boundary_sngrad(BC::RotatingWall, Uc, delta, normal, faces, fID, time) =
    (convert(typeof(Uc), BC.value(faces[fID])) - Uc)/delta
# zero-gradient conditions (Outlet and NeumannFunction are zero gradient for velocity)
@inline _boundary_sngrad(::Union{Zerogradient,Extrapolated,Outlet,NeumannFunction}, Uc, delta, normal, faces, fID, time) =
    zero(Uc)
# mirror condition: face value U_c - (U_c·n)n, normal gradient -(U_c·n)n/delta
@inline _boundary_sngrad(::Union{Slip,Symmetry}, Uc, delta, normal, faces, fID, time) = -(Uc⋅normal)*normal/delta
# conditions that do not apply to velocity (scalar-only ones, vector Neumann is not implemented);
# periodic faces are handled as internal faces by the callers
@inline _boundary_sngrad(BC, Uc, delta, normal, faces, fID, time) = nothing

@inline _patch_gradient(Mc, ::Nothing, normal) = Mc
@inline _patch_gradient(Mc, sngrad, normal) = Mc + (sngrad - Mc*normal)*normal'
