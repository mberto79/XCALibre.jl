export SensibleEnthalpy
export Ttoh, htoT!, Ttoh!, thermo_Psi!

# Model type definition
"""
    SensibleEnthalpy <: AbstractEnergyModel

Type that represents energy model, coefficients and respective fields.

### Fields
- 'h'    -- Sensible enthalpy ScalarField.
- 'T'    -- Temperature ScalarField.
- 'hf'   -- Sensible enthalpy FaceScalarField.
- 'Tf'   -- Temperature FaceScalarField.
- 'K'    -- Specific kinetic energy ScalarField.
- 'dpdt' -- Pressure time derivative ScalarField.
- 'updated_BC' -- Boundary condition function to convert temperature to sensible enthalp on 
                    on a fixed value boudary.
- 'coeffs' -- A tuple of model coefficients.

"""
struct SensibleEnthalpy{S1,S2,F1,F2,S3,S4,F,C} <: AbstractEnergyModel
    h::S1
    T::S2
    hf::F1
    Tf::F2
    K::S3
    dpdt::S4
    update_BC::F
    coeffs::C
end
Adapt.@adapt_structure SensibleEnthalpy

struct Sensible_Enthalpy_Model{E1}
    energy_eqn::E1 
end
Adapt.@adapt_structure Sensible_Enthalpy_Model

# Model API constructor
Energy{SensibleEnthalpy}(; Tref) = begin
    coeffs = (Tref=Tref, other=nothing)
    ARG = typeof(coeffs)
    Energy{SensibleEnthalpy,ARG}(coeffs)
end

# Functor as consturctor
(energy::Energy{EnergyModel, ARG})(mesh, fluid) where {EnergyModel<:SensibleEnthalpy,ARG} = begin
    h = ScalarField(mesh)
    T = ScalarField(mesh)
    hf = FaceScalarField(mesh)
    Tf = FaceScalarField(mesh)
    K = ScalarField(mesh)
    dpdt = ScalarField(mesh)
    update_BC =  return_thingy(EnergyModel, fluid, energy.args.Tref)
    coeffs = energy.args
    SensibleEnthalpy(h, T, hf, Tf, K, dpdt, update_BC, coeffs)
end

return_thingy(::Type{SensibleEnthalpy}, fluid, Tref) = begin
    function Ttoh(T)
        Cp = fluid.cp
        h = Cp.values*(T-Tref)
        return h
    end
end

"""
    initialise(energy::SensibleEnthalpy, model::Physics{T1,F,M,Tu,E,D,BI}, mdotf, rho, peqn, config
    ) where {T1,F,M,Tu,E,D,BI})

Initialisation of energy transport equations.

### Input
- `energy` -- Energy model.
- `model`  -- Physics model defined by user.
- `mdtof`  -- Face mass flow.
- `rho`    -- Density ScalarField.
- `peqn`   -- Pressure equation.
- `config` -- Configuration structure defined by user with solvers, schemes, runtime and 
              hardware structures set.

### Output
- `Sensible_Enthalpy_Model`  -- Energy model struct containing energy equation.

"""
function initialise(
    energy::SensibleEnthalpy, model::Physics{T1,F,M,Tu,E,D,BI}, mdotf, rho, peqn, config
    ) where {T1,F,M,Tu,E,D,BI}

    (; h, T, dpdt) = energy
    (; solvers, schemes, runtime) = config
    mesh = mdotf.mesh
    eqn = peqn.equation
    
    # rho = ScalarField(mesh)
    keff_by_cp = FaceScalarField(mesh)
    divK = ScalarField(mesh)
    dKdt = ScalarField(mesh)

    Ttoh!(model, T, h)

    energy_eqn = (
        Time{schemes.h.time}(rho, h)
        + Divergence{schemes.h.divergence}(mdotf, h) 
        - Laplacian{schemes.h.laplacian}(keff_by_cp, h) 
        == 
        -Source(divK)
        -Source(dKdt)
        +Source(dpdt)
    ) → eqn
    
    # Set up preconditioners
    @reset energy_eqn.preconditioner = set_preconditioner(
                solvers.h.preconditioner, energy_eqn, h.BCs, config)
    
    # preallocating solvers
    @reset energy_eqn.solver = solvers.h.solver(_A(energy_eqn), _b(energy_eqn))

    return Sensible_Enthalpy_Model(energy_eqn)
end


"""
    energy::Sensible_Enthalpy_Model{E1}, model::Physics{T1,F,M,Tu,E,D,BI}, prev, mdotf, rho, mueff, time, config
    ) where {T1,F,M,Tu,E,D,BI,E1}

Run energy transport equations.

### Input
- `energy` -- Energy model.
- `model`  -- Physics model defined by user.
- `prev`   -- Previous energy cell values.
- `mdtof`  -- Face mass flow.
- `rho`    -- Density ScalarField.
- `mueff`  -- Effective viscosity FaceScalarField.
- `time`   --
- `config` -- Configuration structure defined by user with solvers, schemes, runtime and 
              hardware structures set.

"""
function energy!(
    energy::Sensible_Enthalpy_Model{E1}, model::Physics{T1,F,M,Tu,E,D,BI}, prev, mdotf, rho, mueff, time, config
    ) where {T1,F,M,Tu,E,D,BI,E1}

    mesh = model.domain

    (;U) = model.momentum
    (;h, hf, T, K, dpdt) = model.energy
    (;energy_eqn) = energy
    (; solvers, runtime, hardware) = config
    (; iterations, write_interval) = runtime
    (; backend) = hardware

    # rho = get_flux(energy_eqn, 1)
    keff_by_cp = get_flux(energy_eqn, 3)
    divK = get_source(energy_eqn, 1)
    dKdt = get_source(energy_eqn, 2)

    Uf = FaceVectorField(mesh)
    Kf = FaceScalarField(mesh)
    # Kbounded = ScalarField(mesh)
    Pr = model.fluid.Pr

    dt = runtime.dt

    # Pre-allocate auxiliary variables
    TF = _get_float(mesh)
    n_cells = length(mesh.cells)
    prev = zeros(TF, n_cells)
    prev = _convert_array!(prev, backend) 

    volumes = getproperty.(mesh.cells, :volume)

    @. keff_by_cp.values = mueff.values/Pr.values

    @. prev = K.values
    interpolate!(Uf, U, config)
    correct_boundaries!(Uf, U, U.BCs, time, config)
    for i ∈ eachindex(K)
        K.values[i] = 0.5*(U.x.values[i]^2 + U.y.values[i]^2 + U.z.values[i]^2)
    end
    interpolate!(Kf, K, config)
    for i ∈ eachindex(Kf)
        Kf.values[i] = 0.5*(Uf.x.values[i]^2 + Uf.y.values[i]^2 + Uf.z.values[i]^2)
    end
    # correct_face_interpolation!(Kf, K, mdotf) # This forces KE to be upwind, MIGHT NOT BE WORKING
    @. Kf.values *= mdotf.values
    div!(divK, Kf, config)

    if config.schemes.h.time <: SteadyState
        @. dKdt.values = 0.0
    else
        @. dKdt.values = rho.values*(K.values - prev)/dt
    end

    # Set up and solve energy equation
    @. prev = h.values
    discretise!(energy_eqn, h, config)
    apply_boundary_conditions!(energy_eqn, h.BCs, nothing, time, config)
    implicit_relaxation_diagdom!(energy_eqn, h.values, solvers.h.relax, nothing, config)
    update_preconditioner!(energy_eqn.preconditioner, mesh, config)
    solve_system!(energy_eqn, solvers.h, h, nothing, config)

    if ~isempty(solvers.h.limit)
        Tmin = solvers.h.limit[1]; Tmax = solvers.h.limit[2]
        thermoClamp!(model, h, Tmin, Tmax)
    end

    htoT!(model, h, T)
    interpolate!(hf, h, config)
    correct_boundaries!(hf, h, h.BCs, time, config)
end


"""
    thermo_Psi!(model::Physics{T,F,M,Tu,E,D,BI}, Psi::ScalarField) 
    where {T,F<:AbstractCompressible,M,Tu,E,D,BI}

Model updates the value of Psi.

### Input
- `model`  -- Physics model defined by user.
- `Psi`    -- Compressibility factor ScalarField.

### Algorithm
Weakly compressible currently uses the ideal gas equation for establishing the
compressibility factor where ``\\rho = p * \\Psi``. ``\\Psi`` is calculated from the sensible 
enthalpy, reference temperature and fluid model specified ``C_p`` and ``R`` value where 
``R`` is calculated from ``C_p`` and ``\\gamma`` specified in the fluid model.
"""
function thermo_Psi!(
    model::Physics{T,F,M,Tu,E,D,BI}, Psi::ScalarField
    ) where {T,F<:AbstractCompressible,M,Tu,E,D,BI}
    (; coeffs, h) = model.energy
    (; Tref) = coeffs
    Cp = model.fluid.cp; R = model.fluid.R
    @. Psi.values = Cp.values/(R.values*(h.values + Cp.values*Tref))
end

"""
    thermo_Psi!(model::Physics{T,F,M,Tu,E,D,BI}, Psif::FaceScalarField) 
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
    model::Physics{T,F,M,Tu,E,D,BI}, Psif::FaceScalarField, config
    ) where {T,F<:AbstractCompressible,M,Tu,E,D,BI}
    (; coeffs, hf, h) = model.energy
    interpolate!(hf, h, config)
    correct_boundaries!(hf, h, h.BCs, time, config)
    (; Tref) = coeffs
    Cp = model.fluid.cp; R = model.fluid.R
    @. Psif.values = Cp.values/(R.values*(hf.values + Cp.values*Tref))
end

"""
    Ttoh!(model::Physics{T1,F,M,Tu,E,D,BI}, T::ScalarField, h::ScalarField
    ) where {T1,F<:AbstractCompressible,M,Tu,E,D,BI}

Function coverts temperature ScalarField to sensible enthalpy ScalarField.

### Input
- `model`  -- Physics model defined by user.
- `T`      -- Temperature ScalarField.
- `h`      -- Sensible enthalpy ScalarField.
"""
function Ttoh!(
    model::Physics{T1,F,M,Tu,E,D,BI}, T::ScalarField, h::ScalarField
    ) where {T1,F<:AbstractCompressible,M,Tu,E,D,BI}
    (; coeffs) = model.energy
    (; Tref) = coeffs
    Cp = model.fluid.cp
    @. h.values = Cp.values*(T.values-Tref)
end

"""
    htoT!(model::Physics{T1,F,M,Tu,E,D,BI}, h::ScalarField, T::ScalarField
    ) where {T1,F<:AbstractCompressible,M,Tu,E,D,BI}

Function coverts sensible enthalpy ScalarField to temperature ScalarField.

### Input
- `model`  -- Physics model defined by user.
- `h`      -- Sensible enthalpy ScalarField.
- `T`      -- Temperature ScalarField.
"""
function htoT!(
    model::Physics{T1,F,M,Tu,E,D,BI}, h::ScalarField, T::ScalarField
    ) where {T1,F<:AbstractCompressible,M,Tu,E,D,BI}
    (; coeffs) = model.energy
    (; Tref) = coeffs
    Cp = model.fluid.cp
    @. T.values = (h.values/Cp.values) + Tref
end

function thermoClamp!(
    model::Physics{T1,F,M,Tu,E,D,BI}, h::ScalarField, Tmin, Tmax
    ) where {T1,F<:AbstractCompressible,M,Tu,E,D,BI}
    (; coeffs) = model.energy
    (; Tref) = coeffs
    Cp = model.fluid.cp
    hmin = Cp.values*(Tmin-Tref)
    hmax = Cp.values*(Tmax-Tref)
    clamp!(h.values, hmin, hmax)
end

function thermoClamp!(
    model::Physics{T1,F,M,Tu,E,D,BI}, hf::FaceScalarField, Tmin, Tmax
    ) where {T1,F<:AbstractCompressible,M,Tu,E,D,BI}
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