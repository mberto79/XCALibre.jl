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
@kwdef struct GravityState{V<:AbstractVector{<:AbstractFloat},S1,F1} <: AbstractPhysicsProperty
    g::V
    gh::S1
    ghf::F1
end
Adapt.@adapt_structure GravityState

function build_gravityModel(setup::Gravity, mesh)
    gh = ScalarField(mesh)
    ghf = FaceScalarField(mesh)
    F = _get_float(mesh)
    return GravityState(
        g=SVector{3,F}(setup.g),
        gh=gh,
        ghf=ghf
    )
end