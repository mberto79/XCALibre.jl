export Viscosity, ConstantViscosity, SutherlandViscosity
export initialise_viscosity, update_viscosity!

# Generic Viscosity struct to hold viscosity model arguments
struct Viscosity{T,ARG}
    args::ARG
end


function update_viscosity!(fluid, energy, config)
    update_viscosity_cell!(fluid, energy, config)
    update_viscosity_face!(fluid, fluid.visc_model, config)
end

function update_viscosity_cell!(fluid, energy, config)
    (; hardware) = config
    (; backend, workgroup) = hardware
    (; nu, visc_model) = fluid

    ndrange = length(nu)
    kernal! = _update_viscosity_cell!(_setup(backend, workgroup, ndrange)...)
    kernal!(nu, visc_model, fluid, energy, config)
end


# CONSTANT VISCOSITY

struct ConstantViscosity{} <: AbstractViscosityModel
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
    return nu, nuf, ConstantViscosity()
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
    return nu, nuf, ConstantViscosity()
end

@kernel function _update_viscosity_cell!(nu, visc::ConstantViscosity, fluid, energy, config)
end

function update_viscosity_face!(fluid, visc_model::ConstantViscosity, config)
end

# SUTHERLAND VISCOSITY

struct SutherlandViscosity{C} <: AbstractViscosityModel
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
    return nu, nuf, SutherlandViscosity(coeffs)
end

@kernel function _update_viscosity_cell!(nu, visc_model, fluid, energy, config)
    i = @index(Global)

    @uniform begin
        rho_field = fluid.rho
        T_field = energy.T
    end

    @inbounds begin
        nu[i] = visc_model.coeffs.mu_ref * (T_field[i] / visc_model.coeffs.T_ref)^(3/2) * (visc_model.coeffs.T_ref + visc_model.coeffs.S) / (T_field[i] + visc_model.coeffs.S) / rho_field[i]
    end
end

function update_viscosity_face!(fluid, visc_model::SutherlandViscosity, config)
    interpolate!(fluid.nuf, fluid.nu, config)
end
