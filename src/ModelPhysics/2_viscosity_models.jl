export Viscosity, ConstantViscosity, SutherlandViscosity
export initialise_viscosity, update_viscosity!

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
    nu::S
    nuf::FS
end
Adapt.@adapt_structure ConstantViscosity

Viscosity{ConstantViscosity}(; nu) = begin
    coeffs = (nu=nu, other=nothing)
    ARG = typeof(coeffs)
    Viscosity{ConstantViscosity,ARG}(coeffs)
end

initialise_viscosity(nu::ConstantViscosity, mesh) = begin
    backend = _get_backend(mesh)
    float_type = _get_float(mesh)
    n_cells = length(mesh.cells)
    nu = ConstantScalar(nu.args.nu)
    nuf = nu
    ConstantViscosity(nu, nuf)
end

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
    nu::S
    nuf::FS
    coeffs::C
end 
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
    print(nu)
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
