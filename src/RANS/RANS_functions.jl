export StrainRate
export double_inner_product!
export magnitude!, magnitude2!

struct StrainRate{G, GT} <: AbstractTensorField
    gradU::G
    gradUT::GT
end

function (S::StrainRate)(i)
    0.5.*(S.gradU[i] .+ S.gradUT[i])
end

double_inner_product!(
    # s, t1::AbstractTensorField, t2) = 
    s, t0::AbstractTensorField, t2) = 
begin
    sum = 0.0
    for i ∈ eachindex(s)
        t1 = t0[i] .- (1/3)*t0[i]*I
        sum = 0.0
        for j ∈ 1:3
            for k ∈ 1:3
                # sum +=   S[i][j,k]*S[i][k,j]
                # sum +=   t1[i][j,k]*t2[i][j,k]
                # sum +=   t1[j,k]*t2[i][j,k]
                sum +=   t1[j,k]*t2[i][k,j]
            end
        end
        s[i] = sum
    end
end

function magnitude!(magS::ScalarField, S::AbstractTensorField)
    sum = 0.0
    for i ∈ eachindex(magS.values)
        sum = 0.0
        for j ∈ 1:3
            for k ∈ 1:3
                # sum +=   S[i][j,k]*S[i][k,j]
                sum +=   S[i][j,k]*S[i][k,j]
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
        magS.values[i] = sum
    end
end