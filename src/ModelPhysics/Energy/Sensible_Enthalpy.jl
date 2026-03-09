export SensibleEnthalpy
export Ttoh, htoT!, Ttoh!, thermo_Psi!

# Model type definition
"""
    SensibleEnthalpy <: AbstractEnergyModel

Type that represents energy model, coefficients and respective fields.

### Fields
- `h`: Sensible enthalpy ScalarField.
- `T`: Terature ScalarField.
- `hf`: Sensible enthalpy FaceScalarField.
- `Tf`: Temperature FaceScalarField.
- `K`: Specific kinetic energy ScalarField.
- `dpdt`: Pressure time derivative ScalarField.
- `coeffs`: A tuple of model coefficients.

"""
# struct SensibleEnthalpy{S1,S2,F1,F2,S3,S4,F,C} <: AbstractEnergyModel
struct SensibleEnthalpy{S1,S2,F1,F2,S3,S4,C} <: AbstractEnergyModel
    h::S1
    T::S2
    hf::F1
    Tf::F2
    K::S3
    dpdt::S4
    # update_BC::F
    coeffs::C
end
Adapt.@adapt_structure SensibleEnthalpy

struct SensibleEnthalpyModel{E1,State}
    energy_eqn::E1 
    state::State
end
Adapt.@adapt_structure SensibleEnthalpyModel

# Model API constructor
Energy{SensibleEnthalpy}(; Tref) = begin
    coeffs = (Tref=Tref, other=nothing)
    ARG = typeof(coeffs)
    Energy{SensibleEnthalpy,ARG}(coeffs)
end

# Functor as constructor
(energy::Energy{EnergyModel, ARG})(mesh, fluid) where {EnergyModel<:SensibleEnthalpy,ARG} = begin
    h = ScalarField(mesh)
    T = ScalarField(mesh)
    hf = FaceScalarField(mesh)
    Tf = FaceScalarField(mesh)
    K = ScalarField(mesh)
    dpdt = ScalarField(mesh)
    # update_BC =  return_thingy(EnergyModel, fluid, energy.args.Tref)
    coeffs = energy.args
    # SensibleEnthalpy(h, T, hf, Tf, K, dpdt, update_BC, coeffs)
    SensibleEnthalpy(h, T, hf, Tf, K, dpdt, coeffs)
end

"""
    initialise(energy::SensibleEnthalpy, model::Physics{T1,F,SO,M,Tu,E,D,BI}, mdotf, rho, peqn, config
    ) where {T1,F,SO,M,Tu,E,D,BI})

Initialisation of energy transport equations.

# Input
- `energy`: Energy model.
- `model`: Physics model defined by user.
- `mdtof`: Face mass flow.
- `rho`: Density ScalarField.
- `peqn`: Pressure equation.
- `config`: Configuration structure defined by user with solvers, schemes, runtime and 
              hardware structures set.

# Output
- `SensibleEnthalpyModel`: Energy model struct containing energy equation.

"""
function initialise(
    energy::SensibleEnthalpy, model::Physics{T1,F,SO,M,Tu,E,D,BI}, mdotf, rho, peqn, config
    ) where {T1,F,SO,M,Tu,E,D,BI}

    (; h, T, dpdt) = energy
    (; solvers, schemes, runtime, boundaries) = config
    mesh = mdotf.mesh
    eqn = peqn.equation
    
    # rho = ScalarField(mesh)
    keff_by_cp = FaceScalarField(mesh)
    divK = ScalarField(mesh)
    dKdt = ScalarField(mesh)
    Phi = ScalarField(mesh)

    Ttoh!(model, T, h)

    energy_eqn = (
        Time{schemes.h.time}(rho, h)
        + Divergence{schemes.h.divergence}(mdotf, h) 
        - Laplacian{schemes.h.laplacian}(keff_by_cp, h) 
        == 
        - Source(divK)
        - Source(dKdt)
        # + Source(dpdt)
        # + Source(Phi)
    ) → eqn
    
    # Set up preconditioners
    @reset energy_eqn.preconditioner = set_preconditioner(solvers.h.preconditioner, energy_eqn)
    
    # preallocating solvers
    @reset energy_eqn.solver = _workspace(solvers.h.solver, _b(energy_eqn))

    init_residual = (:h, 1.0)
    init_converged = false
    state = ModelState(init_residual, init_converged)

    return SensibleEnthalpyModel(energy_eqn, state)
end


"""
    energy::SensibleEnthalpyModel, model::Physics{T1,F,SO,M,Tu,E,D,BI}, prev, mdotf, rho, mueff, time, config
    ) where {T1,F,SO,M,Tu,E,D,BI,E1}

Run energy transport equations.

# Input
- `energy`: Energy model.
- `model`: Physics model defined by user.
- `prev`: Previous energy cell values.
- `mdtof`: Face mass flow.
- `rho`: Density ScalarField.
- `mueff`: Effective viscosity FaceScalarField.
- `time`: Simulation runtime.
- `config`: Configuration structure defined by user with solvers, schemes, runtime and hardware structures set.

"""
function energy!(
    energy::SensibleEnthalpyModel, model::Physics{T1,F,SO,M,Tu,E,D,BI}, prev, mdotf, gradU, gradP, rho, mueff, time, config
    ) where {T1,F,SO,M,Tu,E,D,BI}

    mesh = model.domain

    (; rho, nu) = model.fluid
    (;U) = model.momentum
    (;h, hf, T, K) = model.energy
    (;energy_eqn, state) = energy
    (; solvers, runtime, hardware, boundaries) = config
    (; iterations, write_interval) = runtime
    (; backend) = hardware

    # rho = get_flux(energy_eqn, 1)
    keff_by_cp = get_flux(energy_eqn, 3)
    divK = get_source(energy_eqn, 1)
    dKdt = get_source(energy_eqn, 2)
    # dpdt = get_source(energy_eqn, 3)
    # Phi = get_source(energy_eqn, 4)

    Uf = FaceVectorField(mesh)
    Kf = FaceScalarField(mesh)
    # Kbounded = ScalarField(mesh)
    Pr = model.fluid.Pr

    dt = runtime.dt[1]

    # Pre-allocate auxiliary variables
    TF = _get_float(mesh)
    n_cells = length(mesh.cells)
    prev = KernelAbstractions.zeros(backend, TF, n_cells) 

    volumes = getproperty.(mesh.cells, :volume)

    @. keff_by_cp.values = mueff.values/Pr.values

    @. prev = K.values
    interpolate!(Uf, U, config)
    correct_boundaries!(Uf, U, boundaries.U, time, config)

    @. K.values = 0.5*(U.x.values^2 + U.y.values^2 + U.z.values^2)
    @. Kf.values = 0.5*(Uf.x.values^2 + Uf.y.values^2 + Uf.z.values^2) # values are correct at at boundary faces since they are taken directlyfrom the velocity vector which was corrected after the face interpolation above.

    interpolate_upwind!(Kf, K, mdotf, config) # only do internal faces

    @. Kf.values *= mdotf.values
    div!(divK, Kf, config)

    if config.schemes.h.time <: SteadyState
        @. dKdt.values = 0.0
    else
        @. dKdt.values = rho.values*(K.values - prev)/dt
    end

    # update the material derivative Dp/Dt 
    # @. dpdt.values = begin
    #     U.x.values*gradP.result.x.values 
    #     + U.y.values*gradP.result.y.values
    #     + U.z.values*gradP.result.z.values
    # end

    # viscous_dissipation!(Phi, nu, rho, gradU, config)

    # Set up and solve energy equation
    @. prev = h.values
    discretise!(energy_eqn, h, config)
    apply_boundary_conditions!(energy_eqn, boundaries.h, nothing, time, config)
    implicit_relaxation_diagdom!(energy_eqn, h.values, solvers.h.relax, nothing, config)
    update_preconditioner!(energy_eqn.preconditioner, mesh, config)
    h_res = solve_system!(energy_eqn, solvers.h, h, nothing, config)

    if !isnothing(solvers.h.limit)
        Tmin = solvers.h.limit[1]; Tmax = solvers.h.limit[2]
        thermoClamp!(model, h, Tmin, Tmax)
    end

    htoT!(model, h, T)
    interpolate!(hf, h, config)
    correct_boundaries!(hf, h, boundaries.h, time, config)

    residuals = (:h, h_res)
    converged = h_res <= solvers.h.convergence
    state.residuals = residuals
    state.converged = converged

    return nothing
end


function viscous_dissipation!(Phi::ScalarField, nu, rho, gradU, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    
    mesh = Phi.mesh
    n_cells = length(mesh.cells)
    
    kernel! = _viscous_dissipation_kernel!(_setup(backend, workgroup, n_cells)...)
    kernel!(Phi.values, nu, rho, gradU.result)
end

@kernel function _viscous_dissipation_kernel!(Phi, nu, rho, gradU)
    i = @index(Global)

    # Extract the velocity gradient tensor for this specific cell
    G = gradU[i]
    
    # 1. Divergence of U (Trace of the gradient tensor)
    divU = G[1,1] + G[2,2] + G[3,3]
    
    # 2. Sum of the squares of the diagonal terms
    diag_terms_sq = G[1,1]^2 + G[2,2]^2 + G[3,3]^2
    
    # 3. Sum of the squares of the symmetric cross terms
    cross_terms_sq = (G[1,2] + G[2,1])^2 + 
                     (G[1,3] + G[3,1])^2 + 
                     (G[2,3] + G[3,2])^2
                     
    # 4. Final Viscous Dissipation Source Term (Phi)
    Phi[i] = nu[i]*rho[i] * (2.0 * diag_terms_sq + cross_terms_sq - (2.0/3.0) * divU^2)
end


"""
    thermo_Psi!(model::Physics{T,F,SO,M,Tu,E,D,BI}, Psi::ScalarField) 
    where {T,F<:AbstractCompressible,M,Tu,E,D,BI}

Model updates the value of Psi.

### Input
- `model`  -- Physics model defined by user.
- `Psi`    -- Compressibility factor ScalarField.

### Algorithm
Weakly compressible currently uses the ideal gas equation for establishing the
compressibility factor where ``\\rho = p * \\Psi``. ``\\Psi`` is calculated from the sensible enthalpy, reference temperature and fluid model specified ``C_p`` and ``R`` value where ``R`` is calculated from ``C_p`` and ``\\gamma`` specified in the fluid model.
"""
function thermo_Psi!(
    model::Physics{T,F,SO,M,Tu,E,D,BI}, Psi::ScalarField
    ) where {T,F<:AbstractCompressible,SO,M,Tu,E,D,BI}
    (; coeffs, h) = model.energy
    (; Tref) = coeffs
    Cp = model.fluid.cp; R = model.fluid.R
    @. Psi.values = Cp.values/(R.values*(h.values + Cp.values*Tref))
end

"""
    thermo_Psi!(model::Physics{T,F,SO,M,Tu,E,D,BI}, Psif::FaceScalarField) 
    where {T,F<:AbstractCompressible,M,Tu,E,D,BI}

Function updates the value of Psi.

### Input
- `model`  -- Physics model defined by user.
- `Psif`    -- Compressibility factor FaceScalarField.

### Algorithm
Weakly compressible currently uses the ideal gas equation for establishing the
compressibility factor where ``\\rho = p * \\Psi``. ``\\Psi`` is calculated from the sensible 
enthalpy, reference temperature and fluid model specified ``C_p`` and ``R`` value where 
``R`` is calculated from ``C_p`` and ``\\gamma`` specified in the fluid model.
"""
function thermo_Psi!(
    model::Physics{T,F,SO,M,Tu,E,D,BI}, Psif::FaceScalarField, config
    ) where {T,F<:AbstractCompressible,SO,M,Tu,E,D,BI}
    (; coeffs, hf, h) = model.energy
    interpolate!(hf, h, config)
    correct_boundaries!(hf, h, config.boundaries.h, time, config)
    (; Tref) = coeffs
    Cp = model.fluid.cp; R = model.fluid.R
    @. Psif.values = Cp.values/(R.values*(hf.values + Cp.values*Tref))
end

"""
    Ttoh!(model::Physics{T1,F,SO,M,Tu,E,D,BI}, T::ScalarField, h::ScalarField
    ) where {T1,F<:AbstractCompressible,M,Tu,E,D,BI}

Function coverts temperature ScalarField to sensible enthalpy ScalarField.

### Input
- `model`  -- Physics model defined by user.
- `T`      -- Temperature ScalarField.
- `h`      -- Sensible enthalpy ScalarField.
"""
function Ttoh!(
    model::Physics{T1,F,SO,M,Tu,E,D,BI}, T::ScalarField, h::ScalarField
    ) where {T1,F<:AbstractCompressible,SO,M,Tu,E,D,BI}
    (; coeffs) = model.energy
    (; Tref) = coeffs
    Cp = model.fluid.cp
    @. h.values = Cp.values*(T.values-Tref)
end

"""
    htoT!(model::Physics{T1,F,SO,M,Tu,E,D,BI}, h::ScalarField, T::ScalarField
    ) where {T1,F<:AbstractCompressible,M,Tu,E,D,BI}

Function coverts sensible enthalpy ScalarField to temperature ScalarField.

### Input
- `model`  -- Physics model defined by user.
- `h`      -- Sensible enthalpy ScalarField.
- `T`      -- Temperature ScalarField.
"""
function htoT!(
    model::Physics{T1,F,SO,M,Tu,E,D,BI}, h::ScalarField, T::ScalarField
    ) where {T1,F<:AbstractCompressible,SO,M,Tu,E,D,BI}
    (; coeffs) = model.energy
    (; Tref) = coeffs
    Cp = model.fluid.cp
    @. T.values = (h.values/Cp.values) + Tref
end

function thermoClamp!(
    model::Physics{T1,F,SO,M,Tu,E,D,BI}, h::ScalarField, Tmin, Tmax
    ) where {T1,F<:AbstractCompressible,SO,M,Tu,E,D,BI}
    (; coeffs) = model.energy
    (; Tref) = coeffs
    Cp = model.fluid.cp
    hmin = Cp.values*(Tmin-Tref)
    hmax = Cp.values*(Tmax-Tref)
    clamp!(h.values, hmin, hmax)
end

function thermoClamp!(
    model::Physics{T1,F,SO,M,Tu,E,D,BI}, hf::FaceScalarField, Tmin, Tmax
    ) where {T1,F<:AbstractCompressible,SO,M,Tu,E,D,BI}
    (; coeffs) = model.energy
    (; Tref) = coeffs
    Cp = model.fluid.cp
    hmin = Cp.values*(Tmin-Tref)
    hmax = Cp.values*(Tmax-Tref)
    clamp!(hf.values, hmin, hmax)
end

function correct_face_interpolation!(phif::FaceScalarField, phi, Uf::FaceScalarField)
    mesh = phif.mesh
    (; faces, cells) = mesh
    for fID ∈ eachindex(faces)
        face = faces[fID]
        (; ownerCells, area, normal) = face
        cID1 = ownerCells[1]
        cID2 = ownerCells[2]
        phi1 = phi[cID1]
        phi2 = phi[cID2]
        flux = Uf[fID]
        if flux >= 0.0
            phif.values[fID] = phi1
        else
            phif.values[fID] = phi2
        end
    end
end


export interpolate_upwind!

## UPWIND SCALAR INTERPOLATION
function interpolate_upwind!(phif::FaceScalarField, phi::ScalarField, mdotf::FaceScalarField, config)
    # Extract values arrays from scalar fields 
    vals = phi.values
    fvals = phif.values
    flux = mdotf.values

    # Extract faces from mesh
    mesh = phif.mesh
    (; cells, faces) = mesh
    
    # Get the number of boundary faces to skip them
    nbfaces = length(mesh.boundary_cellsID)
    
    # Calculate the number of internal faces (our new ndrange)
    internal_faces_count = length(faces) - nbfaces

    # Launch interpolate kernel only for internal faces
    (; hardware) = config
    (; backend, workgroup) = hardware
    
    kernel! = interpolate_upwind_Scalar!(_setup(backend, workgroup, internal_faces_count)...)
    kernel!(fvals, vals, flux, cells, faces, nbfaces)
end

@kernel function interpolate_upwind_Scalar!(fvals, vals, flux, cells, faces, nbfaces)
    # Define index for thread
    t = @index(Global)
    
    # Offset the index to strictly process internal faces
    i = t + nbfaces

    @inbounds begin
        # Deconstruct faces to get ownerCells
        face = faces[i]
        (; ownerCells) = face

        # Get cell indices
        owner1 = ownerCells[1]
        owner2 = ownerCells[2]

        # Get cell-centered values
        phi1 = vals[owner1]
        phi2 = vals[owner2]

        # Upwind logic: 
        # If mass flux is positive, flow is leaving owner1 towards owner2
        if flux[i] > 0.0
            fvals[i] = phi1
        else
            fvals[i] = phi2
        end
    end
end
