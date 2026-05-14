export GravityState
export Gravity
export build_gravityModel

export AbstractPhysicsProperty


abstract type AbstractPhysicsProperty end

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