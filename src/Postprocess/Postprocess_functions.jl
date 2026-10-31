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

On a distributed mesh the call is collective: every rank must make it, and all ranks return the
force on the whole patch.
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

    kernel! = _pressure_force!(backend)
    kernel!(fx, fy, fz, p, rhoref, faces, boundary_cellsID, IDs_range;
        _dynamic_setup(backend, workgroup, n)...)
    KernelAbstractions.synchronize(backend)

    Fp = global_sum(TF[sum(fx), sum(fy), sum(fz)], mesh) # patch faces are split over the ranks
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
    viscous_force(patch::Symbol, model, config; time=0)

Calculate the viscous force vector `[Fx, Fy, Fz]` acting on a given patch/boundary.

# Input arguments

- `patch::Symbol` name of the boundary of interest (as a `Symbol`)
- `model` `Physics` object defining the simulation
- `config` `Configuration` object (provides boundary conditions and hardware backend)

# Optional arguments

- `time` the time at which the velocity condition of the patch is evaluated (default `0`). It
  matters only for a `DirichletFunction` whose function depends on time: pass the solution time
  (the iteration number for steady runs), as the solvers do when they evaluate the condition.

The force of each face is the viscous stress the momentum equation applies there,
`-ρ ν_eff (∇U_b + ∇U_bᵀ - ⅔ tr(∇U_b) I)⋅n A`. `ν_eff = ν + ν_t` uses the wall-function value of
`ν_t` on the face, and `∇U_b` is the owner-cell gradient (the gradient scheme of `U` in
`config.schemes`, `Gauss` if none) with its normal component replaced by the normal gradient of
the velocity condition, `(U_b - U_c)/δ`. This includes the transposed (explicit) stress term of the
momentum equation, so the result is the force the solver applies. `U_b` is the face value of the
velocity boundary condition: the fixed vector of `Wall` and `Dirichlet`, the function of
`DirichletFunction` evaluated at the face centre and `time`, or the rotation velocity of
`RotatingWall`. Zero-gradient conditions have a zero normal gradient, so only the tangential
derivatives of the cell gradient act there. Patches with a `Slip` or `Symmetry` velocity boundary
condition exert no shear, so their viscous force is zero.

On a distributed mesh the call is collective: every rank must make it, and all ranks return the
force on the whole patch.
"""
function viscous_force(patch::Symbol, model, config; time=0)
    mesh = model.domain
    U = model.momentum.U
    nu = model.fluid.nu
    rho = model.fluid.rho
    TF = _get_float(mesh)
    nut = model.turbulence isa Laminar ? ConstantScalar(zero(TF)) : model.turbulence.nutf # wall-face nut (wall funcs)
    (; backend, workgroup) = config.hardware

    ID = boundary_index(model.boundary_info, patch)
    @info "calculating viscous force on patch: $patch at index $ID"
    BC = _shear_patch_bc(config.boundaries.U, ID, patch)
    if BC isa Union{Slip,Symmetry}
        @info "Viscous force on $patch: (0, 0, 0) (Slip/Symmetry patch carries no shear)"
        return zeros(TF, 3)
    end
    IDs_range = get_boundaries(mesh.boundaries)[ID].IDs_range
    (; faces, boundary_cellsID) = mesh

    gradU = _velocity_gradient(U, TF(time), config)

    n = length(IDs_range)
    fx = KernelAbstractions.zeros(backend, TF, n)
    fy = KernelAbstractions.zeros(backend, TF, n)
    fz = KernelAbstractions.zeros(backend, TF, n)

    kernel! = _viscous_force!(backend)
    kernel!(fx, fy, fz, U, gradU, BC, nu, nut, rho, faces, boundary_cellsID, IDs_range, TF(time);
        _dynamic_setup(backend, workgroup, n)...)
    KernelAbstractions.synchronize(backend)

    Fv = global_sum(TF[sum(fx), sum(fy), sum(fz)], mesh)
    @info "Viscous force on $patch: ($(Fv[1]), $(Fv[2]), $(Fv[3]))"
    return Fv
end

@kernel function _viscous_force!(
    fx, fy, fz, U, gradU, BC, nu, nut, rho, faces, boundary_cellsID, IDs_range, time)
    i = @index(Global)
    @inbounds begin
        fID = IDs_range[i]
        cID = boundary_cellsID[fID]
        nueff = nu[cID] + nut[fID] # nut is wall-face value νtf (wall funcs)
        F = rho[cID]*faces.area[fID]*_wall_traction(BC, U[cID], gradU[cID], nueff, faces, fID, time)
        fx[i] = F[1]
        fy[i] = F[2]
        fz[i] = F[3]
    end
end

# Kinematic force per area on face fID, -ν_eff (∇U_b + ∇U_bᵀ - ⅔ tr(∇U_b) I)⋅n: the momentum boundary
# stress (Laplacian plus `_transpose_stress_boundary_kernel!`), ∇U_b being the cell gradient Mc with
# the normal gradient of the velocity condition (`_boundary_sngrad` at `time`) as normal component
@inline function _wall_traction(BC, Uc, Mc, nueff, faces, fID, time)
    normal = faces.normal[fID]
    Mb = _patch_gradient(Mc, _boundary_sngrad(BC, Uc, faces.delta[fID], normal, faces, fID, time), normal)
    TF = typeof(nueff)
    return -nueff*((Mb + Mb' - (TF(2)/3)*tr(Mb)*I)*normal)
end

# Cell gradient of U as the solvers compute it (gradient scheme and limiter of U in
# `config.schemes`, Gauss without a limiter if the configuration has no U schemes); returns
# the mesh-free values for the kernels
function _velocity_gradient(U, time, config)
    schemes = config.schemes
    hasU = schemes isa NamedTuple && haskey(schemes, :U)
    gradU = hasU ? Grad{schemes.U.gradient}(U) : Grad{Gauss}(U)
    grad!(gradU, FaceVectorField(U.mesh), U, config.boundaries.U, time, config)
    hasU && limit_gradient!(schemes.U.limiter, gradU, U, config)
    return field_values(gradU.result)
end

# Host-side lookup of the velocity condition of a boundary ID (BCs is a heterogeneous tuple);
# periodic and empty patches have no wall, so no wall shear
_patch_bc(BCs, ID) = begin
    for bc ∈ BCs
        bc.ID == ID && return bc
    end
    error("No boundary condition found for boundary index $ID")
end
_shear_patch_bc(BCs, ID, patch) = _check_shear_patch(_patch_bc(BCs, ID), patch)
_check_shear_patch(BC, patch) = BC
_check_shear_patch(BC::Union{Periodic,PeriodicParent,Empty}, patch) = throw(ArgumentError(
    "patch $patch has a $(nameof(typeof(BC))) velocity condition, which carries no wall shear"))

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
    wall_shear_stress(patch::Symbol, model, config; time=0)

Function to calculate the wall shear stress acting on a given patch/boundary. Returns the
kinematic viscous stress acting on each face of the patch, `-ν_eff (∇U_b + ∇U_bᵀ - ⅔ tr(∇U_b) I)⋅n`
(the face values of [`viscous_force`](@ref) per unit area and density; zero on `Slip` and
`Symmetry` patches), as a `FaceVectorField` of the patch length, and the face centres as a
vector.

# Input arguments

- `patch::Symbol` name of the boundary of interest (as a `Symbol`)
- `model` instance of `Physics` object needs to be passed 
- `config` need to pass `Configuration` object as this contains the boundary conditions

# Optional arguments

- `time` the time at which the velocity condition of the patch is evaluated (default `0`), as
  in [`viscous_force`](@ref), which uses the same face values.
"""
wall_shear_stress(patch::Symbol, model, config; time=0) = begin
    mesh = model.domain
    TF = _get_float(mesh)
    nut = model.turbulence isa Laminar ? ConstantScalar(zero(TF)) : model.turbulence.nutf # wall-face νtf
    (; nu) = model.fluid
    (; U) = model.momentum
    (; backend, workgroup) = config.hardware
    (; boundary_cellsID, faces) = mesh
    ID = boundary_index(model.boundary_info, patch)
    IDs_range = get_boundaries(mesh.boundaries)[ID].IDs_range
    @info "calculating wall shear stress on patch: $patch at index $ID"
    BC = _shear_patch_bc(config.boundaries.U, ID, patch)

    n = length(IDs_range)
    x = FaceScalarField(KernelAbstractions.zeros(backend, TF, n), mesh)
    y = FaceScalarField(KernelAbstractions.zeros(backend, TF, n), mesh)
    z = FaceScalarField(KernelAbstractions.zeros(backend, TF, n), mesh)
    tauw = FaceVectorField(x, y, z, mesh)

    if !(BC isa Union{Slip,Symmetry}) # mirror conditions carry no shear (as in viscous_force)
        gradU = _velocity_gradient(U, TF(time), config)
        kernel! = _wall_shear_stress!(backend)
        kernel!(tauw, U, gradU, BC, nu, nut, faces, boundary_cellsID, IDs_range, TF(time);
            _dynamic_setup(backend, workgroup, n)...)
        KernelAbstractions.synchronize(backend)
    end

    pos = Array(faces.centre[IDs_range])
    return tauw, pos
end

@kernel function _wall_shear_stress!(
    tauw, U, gradU, BC, nu, nut, faces, boundary_cellsID, IDs_range, time)
    i = @index(Global)
    @inbounds begin
        fID = IDs_range[i]
        cID = boundary_cellsID[fID]
        nueff = nu[cID] + nut[fID] # nut is wall-face value νtf (wall funcs)
        tauw[i] = _wall_traction(BC, U[cID], gradU[cID], nueff, faces, fID, time)
    end
end
"""
    stress_tensor(U::VectorField, ν, νt, config; time=0)

Function to calculate the stress tensor.

# Input arguments

- `U::VectorField` velocity field
- `ν` laminar viscosity of the fluid
- `νt` eddy viscosity from turbulence models. Pass ConstantScalar(0) for laminar flows
- `config` need to pass `Configuration` object as this contains the boundary conditions

# Optional arguments

- `time` the time at which the velocity boundary conditions are evaluated (default `0`), as in
  [`viscous_force`](@ref)
"""
stress_tensor(U, ν, νt, config; time=0) = begin
    mesh = U.mesh
    TF = _get_float(mesh)
    gradU = Grad{Midpoint}(U)
    gradUT = T(gradU)
    Uf = FaceVectorField(U.mesh)
    grad!(gradU, Uf, U, config.boundaries.U, TF(time), config)
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
