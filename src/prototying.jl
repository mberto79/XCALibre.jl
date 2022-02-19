struct SteadyDiffusion{T,I,F}
    laplacian::Laplacian{T}
    sign::Vector{Int64}
    equation::Equation{I,F}
end

struct Laplacian{T}
    ϕ::ScalarField{}