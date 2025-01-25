export inner_product!
export double_inner_product!
export magnitude!, magnitude2!

inner_product!(S::F, ∇1::Grad, ∇2::Grad, config) where F<:ScalarField = begin
    (; hardware) = config
    (; backend, workgroup) = hardware

    kernel! = _inner_product!(backend, workgroup)
    kernel!(S, ∇1, ∇2, ndrange = length(S))
    # KernelAbstractions.synchronize(backend)
end

@kernel function _inner_product!(S::F, ∇1::Grad, ∇2::Grad) where F<:ScalarField
    i = @index(Global)
    @uniform values = S.values
    # for i ∈ eachindex(S.values)
        values[i] = ∇1[i]⋅∇2[i]
    # end
end

double_inner_product!(
    s, t0::AbstractTensorField, t2) = 
begin
    sum = 0.0
    for i ∈ eachindex(s)
        t1 = 2.0.*t0[i] .- (2/3)*t0[i]*I
        sum = 0.0
        for j ∈ 1:3
            for k ∈ 1:3
                sum +=   t1[j,k]*t2[i][k,j]
            end
        end
        s[i] = sum
    end
end

# function magnitude!(magS::ScalarField, S::AbstractVectorField, config)
function magnitude!(magS::ScalarField, S, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    kernel! = _magnitude!(backend, workgroup)
    kernel!(magS, S, ndrange = length(magS))
    # KernelAbstractions.synchronize(backend)
end

# @kernel function _magnitude!(magS::ScalarField, S::AbstractVectorField)
@kernel function _magnitude!(magS::ScalarField, S)
    i = @index(Global)
    @uniform values = magS.values
    
    @inbounds values[i] = norm(S[i])
    # sum = 0.0
    # for i ∈ eachindex(magS.values)
    #     sum = 0.0
    #     for j ∈ 1:3
    #         for k ∈ 1:3
    #             sum +=   S[i][j,k]*S[i][k,j]
    #         end
    #     end
    #     magS.values[i] =   sqrt(sum)
    # end
end

function magnitude2!(
    magS::ScalarField, S::AbstractTensorField, config; scale_factor=1.0
    )
    (; hardware) = config
    (; backend, workgroup) = hardware

    kernel! = _magnitude2!(backend, workgroup)
    kernel!(magS, S, scale_factor, ndrange = length(magS))
    # KernelAbstractions.synchronize(backend)
end

@kernel function _magnitude2!(
    magS::ScalarField, S::AbstractTensorField, scale_factor
    )
    i = @index(Global)

    @uniform values = magS.values

    @inbounds begin
        sum = 0.0
        Sjk = S[i]
        for j ∈ 1:3
            for k ∈ 1:3
                sum +=   Sjk[j,k]*Sjk[j,k]
                # sum +=   S(i)[j,k]*S(i)[k,j]
            end
        end
        magS.values[i] = sum*scale_factor
    end
end
