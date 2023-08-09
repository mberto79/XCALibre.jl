export StrainRate
export magnitude!, magnitude2!

struct StrainRate{G, GT} <: AbstractTensorField
    gradU::G
    gradUT::GT
end

function (S::StrainRate)(i)
    0.5.*(S.gradU[i] .+ S.gradUT[i])
end

function magnitude!(magS::ScalarField, S::AbstractTensorField)
    sum = 0.0
    for i ∈ eachindex(magS.values)
        sum = 0.0
        for j ∈ 1:3
            for k ∈ 1:3
                # sum +=   S[i][j,k]*S[i][k,j]
                sum +=   S[i][j,k]*S[i][j,k]
            end
        end
        magS.values[i] =   sqrt(sum)
    end
end

function magnitude2!(magS::ScalarField, S::AbstractTensorField)
    sum = 0.0
    for i ∈ eachindex(magS.values)
        sum = 0.0
        for j ∈ 1:3
            for k ∈ 1:3
                # sum +=   S[i][j,k]*S[i][k,j]
                sum +=   S[i][j,k]*S[i][j,k]
            end
        end
        magS.values[i] =   sum
    end
end