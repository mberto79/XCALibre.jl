# export Laplacian, SteadyDiffusion

# struct Laplacian{T,I,F} <: AbstractTerm 
#     Γ::F 
#     ϕ::ScalarField{I,F}
#     distretisation::T
# end
# Laplacian{Linear}(Γ, ϕ) = Laplacian(Γ, ϕ, Linear())

# struct SteadyDiffusion{T,I,F}
#     laplacian::Laplacian{T}
#     sign::Vector{I}
#     equation::Equation
# end