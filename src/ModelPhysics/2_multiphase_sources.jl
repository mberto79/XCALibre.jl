export LeeModelState, DriftVelocityState, GravityState, CSF_State, ArtificialCompressionState
export LeeModel, DriftVelocity, Gravity, CSF, ArtificialCompression
export build_gravityModel, build_leeModel, build_driftVelocity, build_CSF_Model, build_ArtificialCompressionModel
export AbstractPhysicsProperty

export update_source


abstract type AbstractPhysicsProperty end

abstract type AbstractDrag <: AbstractPhysicsProperty end


Base.@kwdef struct Drag_SchillerNaumann <: AbstractDrag end # not actually used yet but would be nice to define drag models in the future

Base.@kwdef struct Gravity{V<:AbstractVector{<:AbstractFloat}} <: AbstractPhysicsProperty
    g::V
end
@kwdef struct GravityState{V<:AbstractVector{<:AbstractFloat}} <: AbstractPhysicsProperty
    g::V
end
Adapt.@adapt_structure GravityState

function build_gravityModel(setup::Gravity, mesh)
    return GravityState(
        g=setup.g
    )
end
function update_source(model_specific::AbstractFluid, source::GravityState, model, alpha, rho, phases, config, mesh, time) # alpha eqn
    return ConstantScalar(0.0), 1
end
function update_source(model_specific::AbstractEnergyModel, source::GravityState, model, alpha, rho, phases, config, mesh, time) # energy eqn
    return ConstantScalar(0.0), 1
end
function update_source(model_specific::AbstractMomentumModel, source::GravityState, model, alpha, rho, phases, config, mesh, time) # momentum eqn
    g = source.g

    x0, y0, z0 = g[1], g[2], g[3]
    
    x = ScalarField(mesh)
    y = ScalarField(mesh)
    z = ScalarField(mesh)
    initialise!(x, x0)
    initialise!(y, y0)
    initialise!(z, z0)

    rho_ref = ConstantScalar(1.225) # Prototyping
    # rho_ref = min(phase1, phase2)

    # @. x.values = x.values * (rho.values - rho_ref.values)
    # @. y.values = y.values * (rho.values - rho_ref.values)
    # @. z.values = z.values * (rho.values - rho_ref.values)

    @. x.values = x.values * rho.values
    @. y.values = y.values * rho.values
    @. z.values = z.values * rho.values

    rhoG = VectorField(x, y, z, mesh)
    
    return rhoG, 1 # Source, sign
end


Base.@kwdef struct CSF{T<:AbstractFloat} <: AbstractPhysicsProperty
    sigma::T
end
@kwdef struct CSF_State{T<:AbstractFloat} <: AbstractPhysicsProperty
    sigma::T
end
Adapt.@adapt_structure CSF_State

function build_CSF_Model(setup::CSF, mesh)
    return CSF_State(
        sigma=setup.sigma
    )
end
function update_source(model_specific::AbstractFluid, source::CSF_State, model, alpha, rho, phases, config, mesh, time) # alpha eqn
    return ConstantScalar(0.0), 1
end
function update_source(model_specific::AbstractEnergyModel, source::CSF_State, model, alpha, rho, phases, config, mesh, time) # energy eqn
    return ConstantScalar(0.0), 1
end
function update_source(model_specific::AbstractMomentumModel, source::CSF_State, model, alpha, rho, phases, config, mesh, time) # momentum eqn
    (; solvers, schemes, runtime, hardware, boundaries) = config
    
    backend = config.hardware.backend
    workgroup = config.hardware.workgroup

    alphaf = model.fluid.alphaf
    sigma = source.sigma
    
    ∇alpha = Grad{schemes.alpha.gradient}(alpha)
    grad!(∇alpha, alphaf, alpha, boundaries.alpha, time, config)
    limit_gradient!(schemes.alpha.limiter, ∇alpha, alpha, config)

    ST_force = VectorField(mesh)
    n_hat = FaceVectorField(mesh)
    kappa = ScalarField(mesh)

    ndrange = length(∇alpha.result)
    kernel! = compute_nhat!(_setup(backend, workgroup, ndrange)...)
    kernel!(n_hat, ∇alpha.result)

    div!(kappa, n_hat, config)
    @. kappa.values = -kappa.values

    ndrange = length(ST_force)
    kernel! = compute_SF!(_setup(backend, workgroup, ndrange)...)
    kernel!(ST_force, sigma, kappa, ∇alpha.result)
    
    return ST_force, 1 # Source, sign
end

@kernel inbounds=true function compute_nhat!(n_hat, ∇alpha)
    i = @index(Global)

    n_hat[i] = ∇alpha[i] / (norm(∇alpha[i]) + eps())
end
@kernel inbounds=true function compute_SF!(ST_force, sigma, kappa, ∇alpha)
    i = @index(Global)

    ST_force[i] = sigma * kappa[i] * ∇alpha[i]
end


Base.@kwdef struct ArtificialCompression{T<:AbstractFloat} <: AbstractPhysicsProperty
    compression_coeff::T #typically 1.0
end
@kwdef struct ArtificialCompressionState{T<:AbstractFloat} <: AbstractPhysicsProperty
    compression_coeff::T
end
Adapt.@adapt_structure ArtificialCompressionState

function build_ArtificialCompressionModel(setup::ArtificialCompression, mesh)
    return ArtificialCompressionState(
        compression_coeff=setup.compression_coeff
    )
end
function update_source(model_specific::AbstractFluid, source::ArtificialCompressionState, model, alpha, rho, phases, config, mesh, time) # alpha eqn
(; solvers, schemes, runtime, hardware, boundaries) = config
    
    backend = config.hardware.backend
    workgroup = config.hardware.workgroup

    alphaf = model.fluid.alphaf
    Uf = model.momentum.Uf
    compression_coeff = source.compression_coeff
    
    ∇alpha = Grad{schemes.alpha.gradient}(alpha)
    grad!(∇alpha, alphaf, alpha, boundaries.alpha, time, config)
    limit_gradient!(schemes.alpha.limiter, ∇alpha, alpha, config)

    U_r = FaceVectorField(mesh)
    n_hat = FaceVectorField(mesh)
    res_term = ScalarField(mesh)

    ndrange = length(∇alpha.result)
    kernel! = compute_nhat!(_setup(backend, workgroup, ndrange)...)
    kernel!(n_hat, ∇alpha.result)

    ndrange = length(U_r)
    kernel! = compute_Ur!(_setup(backend, workgroup, ndrange)...)
    kernel!(U_r, compression_coeff, n_hat, Uf, alphaf)

    div!(res_term, U_r, config)

    return res_term, 1
end
function update_source(model_specific::AbstractEnergyModel, source::ArtificialCompressionState, model, alpha, rho, phases, config, mesh, time) # energy eqn
    return ConstantScalar(0.0), 1
end
function update_source(model_specific::AbstractMomentumModel, source::ArtificialCompressionState, model, alpha, rho, phases, config, mesh, time) # momentum eqn
    dummy_field = VectorField(mesh) # Poor solution, bad constantVector is not possible currently
    return dummy_field, 1
end
@kernel inbounds=true function compute_Ur!(U_r, compression_coeff, n_hat, Uf, alphaf) #This is not pure U_r, but already blended with alpha
    i = @index(Global)

    U_r[i] = compression_coeff * norm(Uf[i]) * n_hat[i] # pure U_r

    U_r[i] = alphaf[i] * (1.0 - alphaf[i]) * U_r[i] # blended with alpha
end


Base.@kwdef struct LeeModel{T<:AbstractFloat} <: AbstractPhysicsProperty
    evap_coeff::T
    condens_coeff::T
end
@kwdef struct LeeModelState{T<:AbstractFloat, M1,M2,LH} <: AbstractPhysicsProperty
    evap_coeff::T
    condens_coeff::T
    m_qp::M1
    m_pq::M2
    latentHeat::LH
end
Adapt.@adapt_structure LeeModelState

function build_leeModel(setup::LeeModel, mesh)
    m_qp        = ScalarField(mesh)
    m_pq        = ScalarField(mesh)
    latentHeat  = ScalarField(mesh)

    return LeeModelState(
        evap_coeff=setup.evap_coeff,
        condens_coeff=setup.condens_coeff,
        m_qp=m_qp,
        m_pq=m_pq,
        latentHeat=latentHeat
    )
end


function update_source(model_specific::AbstractFluid, source::LeeModelState, model, alpha, rho, phases, config, mesh, time) # alpha eqn
    m_qp = source.m_qp
    m_pq = source.m_pq

    leeField = ScalarField(mesh)

    @. leeField.values = m_qp.values - m_pq.values

    return leeField, 1
end
function update_source(model_specific::AbstractEnergyModel, source::LeeModelState, model, alpha, rho, phases, config, mesh, time) # energy eqn
    m_qp = source.m_qp
    m_pq = source.m_pq
    latentHeat = source.latentHeat

    S_h = ScalarField(mesh)
    @. S_h.values = latentHeat.values * (m_qp.values - m_pq.values)

    return S_h, 1
end
function update_source(model_specific::AbstractMomentumModel, source::LeeModelState, model, alpha, rho, phases, config, mesh, time) # momentum eqn
    dummy_field = VectorField(mesh) # Poor solution, bad constantVector is not possible currently
    return dummy_field, 1
end





Base.@kwdef struct DriftVelocity{G<:Gravity, T<:AbstractFloat, D<:AbstractDrag} <: AbstractPhysicsProperty
    gravity::G      # required for acceleration field computation
    d_p::T          # d_p, either computed from subgrid model or defined as const
    drag::D         # Drag_SchillerNaumann for now
end
@kwdef struct DriftVelocityState{G<:Gravity, T<:AbstractFloat, D<:AbstractDrag, V1,V2,V3,V4,V5,V6,V7} <: AbstractPhysicsProperty
    gravity::G 
    d_p::T
    drag::D

    v_dr_p::V1
    v_dr_q::V2
    v_p::V3
    v_q::V4

    v_pq::V5
    v_pq_prev::V6
    U_prev::V7
end
Adapt.@adapt_structure DriftVelocityState

function build_driftVelocity(setup::DriftVelocity, mesh)
    v_dr_p  = VectorField(mesh)
    v_dr_q  = VectorField(mesh)
    v_p     = VectorField(mesh)
    v_q     = VectorField(mesh)

    v_pq      = VectorField(mesh)
    v_pq_prev = VectorField(mesh)
    U_prev    = VectorField(mesh)

    return DriftVelocityState(
        gravity=setup.gravity,
        d_p=setup.d_p,
        drag=setup.drag,
        v_dr_p=v_dr_p,
        v_dr_q=v_dr_q,
        v_pq=v_pq,
        v_pq_prev=v_pq_prev,
        U_prev=U_prev,
        v_p=v_p,
        v_q=v_q
    )
end
function update_source(model_specific::AbstractFluid, source::DriftVelocityState, model, alpha, rho, phases, config, mesh, time) # alpha eqn
    backend = config.hardware.backend
    workgroup = config.hardware.workgroup

    slipVelocityModel = model.fluid.physics_properties.driftVelocity

    vdr_p = slipVelocityModel.v_dr_p
    rho_p = phases[1].rho
    rho_q = phases[2].rho

    alpha = alpha.values
    rho_p = rho_p.values

    div_term = ScalarField(mesh)

    inside_divergence = VectorField(mesh)

    ndrange = length(inside_divergence)
    kernel! = _alpha_driftVelocity!(_setup(backend, workgroup, ndrange)...)
    kernel!(inside_divergence, vdr_p, alpha, rho_p)
    
    div_interpolated = FaceVectorField(mesh)
    interpolate!(div_interpolated, inside_divergence, config)
    div!(div_term, div_interpolated, config)

    return div_term, 1
end
function update_source(model_specific::AbstractEnergyModel, source::DriftVelocityState, model, alpha, rho, phases, config, mesh, time) # energy eqn
    return ConstantScalar(0.0), 1
end
function update_source(model_specific::AbstractMomentumModel, source::DriftVelocityState, model, alpha, rho, phases, config, mesh, time) # momentum eqn
    backend = config.hardware.backend
    workgroup = config.hardware.workgroup

    slipVelocityModel = model.fluid.physics_properties.driftVelocity

    (vdr_p, vdr_q) = (slipVelocityModel.v_dr_p, slipVelocityModel.v_dr_q)

    rho_p = phases[1].rho
    rho_q = phases[2].rho

    alpha = alpha.values
    rho_p = rho_p.values
    rho_q = rho_q.values
    # prob would make sense to put drift velocities inside DriftVelocity() object

    slipVelocity = VectorField(mesh)
    slipVelocityTensor = TensorField(mesh)

    ndrange = length(slipVelocityTensor)
    kernel! = _momentum_driftVelocity!(_setup(backend, workgroup, ndrange)...)
    kernel!(slipVelocityTensor, vdr_p, vdr_q, alpha, rho_p, rho_q)

    div_tensor!(slipVelocityTensor, slipVelocity, mesh, config)

    return slipVelocity, -1 # Source, sign
end


@kernel inbounds=true function _alpha_driftVelocity!(inside_divergence, vdr_p, alpha, rho_p)
    i = @index(Global)

    inside_divergence[i] = vdr_p[i] * alpha[i] * rho_p[i]
end
@kernel inbounds=true function _momentum_driftVelocity!(slipVelocityTensor, vdr_p, vdr_q, alpha, rho_p, rho_q)
    i = @index(Global)

    cross_p = vdr_p[i] * vdr_p[i]'
    cross_q = vdr_q[i] * vdr_q[i]'
    slipVelocityTensor[i] = (alpha[i] * rho_p[i] * cross_p) + ((1.0-alpha[i]) * rho_q[i] * cross_q) #must be cross products
end



function div_tensor!(tensor_field, output_vector, mesh, config)
    resX = ScalarField(mesh)
    resY = ScalarField(mesh)
    resZ = ScalarField(mesh)

    r1 = VectorField(mesh)
    r2 = VectorField(mesh)
    r3 = VectorField(mesh)

    r1f = FaceVectorField(mesh)
    r2f = FaceVectorField(mesh)
    r3f = FaceVectorField(mesh)

    r1.x.values .= tensor_field.xx.values
    r1.y.values .= tensor_field.xy.values
    r1.z.values .= tensor_field.xz.values

    r2.x.values .= tensor_field.yx.values
    r2.y.values .= tensor_field.yy.values
    r2.z.values .= tensor_field.yz.values

    r3.x.values .= tensor_field.zx.values
    r3.y.values .= tensor_field.zy.values
    r3.z.values .= tensor_field.zz.values
    
    interpolate!(r1f, r1, config)
    interpolate!(r2f, r2, config)
    interpolate!(r3f, r3, config)

    div!(resX, r1f, config)
    div!(resY, r2f, config)
    div!(resZ, r3f, config)

    # println(output_vector)
    # println(output_vector)
    output_vector.x.values .= resX.values
    output_vector.y.values .= resY.values
    output_vector.z.values .= resZ.values

    # nothing
end