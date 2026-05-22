export Viscosity, ConstantViscosity, SutherlandViscosity
export initialise_viscosity, update_viscosity!

# Generic Viscosity struct to hold viscosity model arguments
struct Viscosity{T,ARG}
    args::ARG
end


function update_viscosity!(fluid, energy, config)
    update_viscosity_cell!(fluid, energy, config)
    update_viscosity_face!(fluid.nu, config)
end

function update_viscosity_cell!(fluid, energy, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    (; nu) = fluid.nu

    ndrange = length(nu)
    kernal! = _update_viscosity_cell!(_setup(backend, workgroup, ndrange)...)
    kernal!(nu, fluid.nu, fluid, energy, config)
end


# CONSTANT VISCOSITY

struct ConstantViscosity{S,FS} <: AbstractViscosityModel
    nu::S  # Kinematic viscosity as a scalar field
    nuf::FS  # Face kinematic viscosity as a scalar field
end
Adapt.@adapt_structure ConstantViscosity

Viscosity{ConstantViscosity}(; nu) = begin
    coeffs = (nu=nu, other=nothing)
    ARG = typeof(coeffs)
    Viscosity{ConstantViscosity,ARG}(coeffs)
end

# Function to initialize constant viscosity model from a ConstatViscosity input
# Arguments:
# - nu: Kinematic viscosity as ConstantViscosity model
# - mesh: Computational mesh
# Returns:
# - Initialized ConstantViscosity object with scalar fields
initialise_viscosity(nu::ConstantViscosity, mesh) = begin
    backend = _get_backend(mesh)
    float_type = _get_float(mesh)
    n_cells = length(mesh.cells)
    nu = ConstantScalar(nu.args.nu)
    nuf = nu
    ConstantViscosity(nu, nuf)
end

# Function to initialize constant viscosity model from a Float64 value
# Arguments:
# - nu: Kinematic viscosity as Float64
# - mesh: Computational mesh
# Returns:
# - Initialized ConstantViscosity object with scalar fields
initialise_viscosity(nu::Float64, mesh) = begin
    backend = _get_backend(mesh)
    float_type = _get_float(mesh)
    n_cells = length(mesh.cells)
    nu = ConstantScalar(nu)
    nuf = nu
    ConstantViscosity(nu, nuf)
end

@kernel function _update_viscosity_cell!(nu, visc::ConstantViscosity, fluid, energy, config)
end

function update_viscosity_face!(nu::ConstantViscosity, config)
end

# SUTHERLAND VISCOSITY

struct SutherlandViscosity{S,FS,C} <: AbstractViscosityModel
    nu::S  # Kinematic viscosity as a scalar field
    nuf::FS  # Face kinematic viscosity as a scalar field
    coeffs::C  # Coefficients for the Sutherland model
end

# Adapt structure for GPU compatibility
Adapt.@adapt_structure SutherlandViscosity

Viscosity{SutherlandViscosity}(; mu_ref, T_ref, S) = begin
    coeffs = (mu_ref=mu_ref, T_ref=T_ref, S=S, other=nothing)
    ARG = typeof(coeffs)
    Viscosity{SutherlandViscosity,ARG}(coeffs)
end

initialise_viscosity(nu::Viscosity{SutherlandViscosity}, mesh) = begin
    backend = _get_backend(mesh)
    float_type = _get_float(mesh)
    n_cells = length(mesh.cells)
    coeffs = nu.args
    nu = ScalarField(mesh)
    nuf = FaceScalarField(mesh)
    SutherlandViscosity(nu, nuf, coeffs)
end

@kernel function _update_viscosity_cell!(nu, visc::SutherlandViscosity, fluid, energy, config)
    i = @index(Global)

    @uniform begin
        rho_field = fluid.rho
        T_field = energy.T
    end

    @inbounds begin
        nu[i] = visc.coeffs.mu_ref * (T_field[i] / visc.coeffs.T_ref)^(3/2) * (visc.coeffs.T_ref + visc.coeffs.S) / (T_field[i] + visc.coeffs.S) / rho_field[i]
    end
end

function update_viscosity_face!(nu::SutherlandViscosity, config)
    interpolate!(nu.nuf, nu.nu, config)
end
