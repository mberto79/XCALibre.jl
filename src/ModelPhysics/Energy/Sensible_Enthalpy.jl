export SensibleEnthalpy
export temperature_to_energy!, energy_to_temperature!, thermo_Psi!, energy_clamp!

# Model type definition
"""
    SensibleEnthalpy <: AbstractEnergyModel

Type that represents energy model, coefficients and respective fields.

### Fields
- `he`: Sensible enthalpy ScalarField.
- `T`: Temperature ScalarField.
- `hef`: Sensible enthalpy FaceScalarField.
- `Tf`: Temperature FaceScalarField.
- `K`: Specific kinetic energy ScalarField.
- `S_he`: Energy source term ScalarField (dp/dt for enthalpy).
- `coeffs`: A tuple of model coefficients.

"""
struct SensibleEnthalpy{S1,S2,F1,F2,S3,S4,C} <: AbstractEnergyModel
    he::S1
    T::S2
    hef::F1
    Tf::F2
    K::S3
    S_he::S4
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
    he = ScalarField(mesh)
    T = ScalarField(mesh)
    hef = FaceScalarField(mesh)
    Tf = FaceScalarField(mesh)
    K = ScalarField(mesh)
    S_he = ScalarField(mesh)
    coeffs = energy.args
    SensibleEnthalpy(he, T, hef, Tf, K, S_he, coeffs)
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

    (; he, T, S_he) = energy
    (; solvers, schemes, runtime, boundaries) = config
    mesh = mdotf.mesh
    eqn = peqn.equation

    keff_by_cp = FaceScalarField(mesh)
    divK = ScalarField(mesh)
    dKdt = ScalarField(mesh)

    temperature_to_energy!(model, T, he)

    energy_eqn = (
        Time{schemes.he.time}(rho, he)
        + Divergence{schemes.he.divergence}(mdotf, he)
        - Laplacian{schemes.he.laplacian}(keff_by_cp, he)
        ==
        Source(S_he) - Source(divK) - Source(dKdt)
    ) → eqn

    # Set up preconditioners
    @reset energy_eqn.preconditioner = set_preconditioner(solvers.he.preconditioner, energy_eqn)

    # preallocating solvers
    @reset energy_eqn.solver = _workspace(solvers.he.solver, _b(energy_eqn))

    init_residual = (:he, 1.0)
    init_converged = false
    state = ModelState(init_residual, init_converged)

    return SensibleEnthalpyModel(energy_eqn, state)
end


"""
    energy!(energy::SensibleEnthalpyModel, model::Physics{T1,F,SO,M,Tu,E,D,BI}, prev, mdotf, rho, mueff, time, config
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
    energy::SensibleEnthalpyModel, model::Physics{T1,F,SO,M,Tu,E,D,BI}, prevP, prevRhoK, mdotf, gradP, gradU, rho, mueff, time, config
    ) where {T1,F,SO,M,Tu,E,D,BI}

    mesh = model.domain

    (; rho, nu) = model.fluid
    (;U, p) = model.momentum
    (;he, hef, T, K) = model.energy
    (;energy_eqn, state) = energy
    (; solvers, runtime, hardware, boundaries) = config
    (; iterations, write_interval) = runtime
    (; backend) = hardware

    keff_by_cp = get_flux(energy_eqn, 3)

    S_he = get_source(energy_eqn, 1)
    divK = get_source(energy_eqn, 2)
    dKdt = get_source(energy_eqn, 3)

    Uf = FaceVectorField(mesh)
    Kf = FaceScalarField(mesh)
    Pr = model.fluid.Pr

    dt = runtime.dt[1]

    # Pre-allocate auxiliary variables
    TF = _get_float(mesh)
    n_cells = length(mesh.cells)

    volumes = getproperty.(mesh.cells, :volume)

    @. keff_by_cp.values = mueff.values/Pr.values

    interpolate!(Uf, U, config)
    correct_boundaries!(Uf, U, boundaries.U, time, config)

    @. K.values = 0.5*(U.x.values^2 + U.y.values^2 + U.z.values^2)
    @. Kf.values = 0.5*(Uf.x.values^2 + Uf.y.values^2 + Uf.z.values^2)

    interpolate_upwind!(Kf, K, mdotf, config) # only do internal faces

    @. Kf.values *= mdotf.values
    div!(divK, Kf, config)

    @. dKdt.values = (rho.values*K.values - prevRhoK)/dt
    @. S_he.values = (p.values - prevP)/dt

    # Set up and solve energy equation
    discretise!(energy_eqn, he, config)
    apply_boundary_conditions!(energy_eqn, boundaries.he, nothing, time, config)
    implicit_relaxation_diagdom!(energy_eqn, he.values, solvers.he.relax, nothing, config)
    update_preconditioner!(energy_eqn.preconditioner, mesh, config)
    he_res = solve_system!(energy_eqn, solvers.he, he, nothing, config)

    if !isnothing(solvers.he.limit)
        Tmin = solvers.he.limit[1]; Tmax = solvers.he.limit[2]
        energy_clamp!(model, he, Tmin, Tmax)
    end

    energy_to_temperature!(model, he, T)
    interpolate!(hef, he, config)
    correct_boundaries!(hef, he, boundaries.he, time, config)

    @. prevRhoK = rho.values*0.5*(U.x.values^2 + U.y.values^2 + U.z.values^2)
    @. prevP = p.values

    residuals = (:he, he_res)
    converged = he_res <= solvers.he.convergence
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
    where {T,F<:AbstractCompressible,M,Tu,E<:SensibleEnthalpy,D,BI}

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
    ) where {T,F<:AbstractCompressible,SO,M,Tu,E<:SensibleEnthalpy,D,BI}
    (; coeffs, he) = model.energy
    (; Tref) = coeffs
    Cp = model.fluid.cp; R = model.fluid.R
    @. Psi.values = Cp.values/(R.values*(he.values + Cp.values*Tref))
end

"""
    thermo_Psi!(model::Physics{T,F,SO,M,Tu,E,D,BI}, Psif::FaceScalarField)
    where {T,F<:AbstractCompressible,M,Tu,E<:SensibleEnthalpy,D,BI}

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
    ) where {T,F<:AbstractCompressible,SO,M,Tu,E<:SensibleEnthalpy,D,BI}
    (; coeffs, hef, he) = model.energy
    interpolate!(hef, he, config)
    correct_boundaries!(hef, he, config.boundaries.he, time, config)
    (; Tref) = coeffs
    Cp = model.fluid.cp; R = model.fluid.R
    @. Psif.values = Cp.values/(R.values*(hef.values + Cp.values*Tref))
end

"""
    temperature_to_energy!(model::Physics{T1,F,SO,M,Tu,E,D,BI}, T::ScalarField, he::ScalarField
    ) where {T1,F<:AbstractCompressible,M,Tu,E<:SensibleEnthalpy,D,BI}

Function converts temperature ScalarField to sensible enthalpy ScalarField.

### Input
- `model`  -- Physics model defined by user.
- `T`      -- Temperature ScalarField.
- `he`     -- Sensible enthalpy ScalarField.
"""
function temperature_to_energy!(
    model::Physics{T1,F,SO,M,Tu,E,D,BI}, T::ScalarField, he::ScalarField
    ) where {T1,F<:AbstractCompressible,SO,M,Tu,E<:SensibleEnthalpy,D,BI}
    (; coeffs) = model.energy
    (; Tref) = coeffs
    Cp = model.fluid.cp
    @. he.values = Cp.values*(T.values-Tref)
end

"""
    energy_to_temperature!(model::Physics{T1,F,SO,M,Tu,E,D,BI}, he::ScalarField, T::ScalarField
    ) where {T1,F<:AbstractCompressible,M,Tu,E<:SensibleEnthalpy,D,BI}

Function converts sensible enthalpy ScalarField to temperature ScalarField.

### Input
- `model`  -- Physics model defined by user.
- `he`     -- Sensible enthalpy ScalarField.
- `T`      -- Temperature ScalarField.
"""
function energy_to_temperature!(
    model::Physics{T1,F,SO,M,Tu,E,D,BI}, he::ScalarField, T::ScalarField
    ) where {T1,F<:AbstractCompressible,SO,M,Tu,E<:SensibleEnthalpy,D,BI}
    (; coeffs) = model.energy
    (; Tref) = coeffs
    Cp = model.fluid.cp
    @. T.values = (he.values/Cp.values) + Tref
end

function energy_clamp!(
    model::Physics{T1,F,SO,M,Tu,E,D,BI}, he::ScalarField, Tmin, Tmax
    ) where {T1,F<:AbstractCompressible,SO,M,Tu,E<:SensibleEnthalpy,D,BI}
    (; coeffs) = model.energy
    (; Tref) = coeffs
    Cp = model.fluid.cp
    hmin = Cp.values*(Tmin-Tref)
    hmax = Cp.values*(Tmax-Tref)
    clamp!(he.values, hmin, hmax)
end

function energy_clamp!(
    model::Physics{T1,F,SO,M,Tu,E,D,BI}, hef::FaceScalarField, Tmin, Tmax
    ) where {T1,F<:AbstractCompressible,SO,M,Tu,E<:SensibleEnthalpy,D,BI}
    (; coeffs) = model.energy
    (; Tref) = coeffs
    Cp = model.fluid.cp
    hmin = Cp.values*(Tmin-Tref)
    hmax = Cp.values*(Tmax-Tref)
    clamp!(hef.values, hmin, hmax)
end

function correct_face_interpolation!(phif::FaceScalarField, phi, Uf::FaceVectorField)
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

# UPWIND SCALAR INTERPOLATION
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
        if flux[i] >= 0.0
            fvals[i] = phi1
        else
            fvals[i] = phi2
        end
    end
end
