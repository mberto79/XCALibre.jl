export StrainRate
export double_inner_product!
export magnitude!, magnitude2!
export number_symbols

struct StrainRate{G, GT} <: AbstractTensorField
    gradU::G
    gradUT::GT
end
Adapt.@adapt_structure StrainRate

function (S::StrainRate)(i)
    0.5.*(S.gradU[i] .+ S.gradUT[i])
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

function magnitude!(magS::ScalarField, S::AbstractTensorField)
    sum = 0.0
    for i ∈ eachindex(magS.values)
        sum = 0.0
        for j ∈ 1:3
            for k ∈ 1:3
                sum +=   S[i][j,k]*S[i][k,j]
            end
        end
        magS.values[i] =   sqrt(sum)
    end
end

function magnitude2!(
    magS::ScalarField, S::AbstractTensorField, config; scale_factor=1.0
    )
    (; hardware) = config
    (; backend, workgroup) = hardware

    # Launch interpolate midpoint kernel for scalar field
    kernel! = _magnitude2!(backend, workgroup)
    kernel!(magS, S, scale_factor, ndrange = length(magS))
    KernelAbstractions.synchronize(backend)
end

@kernel function _magnitude2!(
    magS::ScalarField, S::AbstractTensorField, scale_factor
    )
    i = @index(Global)

    @uniform values = magS.values

    @inbounds begin
        sum = 0.0
        for j ∈ 1:3
            for k ∈ 1:3
                sum +=   S[i][j,k]*S[i][j,k]
            end
        end
        magS.values[i] = sum*scale_factor
    end
end

function bound!(field, config)
    # Extract hardware configuration
    (; hardware) = config
    (; backend, workgroup) = hardware

    (; values, mesh) = field
    (; cells, cell_neighbours) = mesh

    # set up and launch kernel
    kernel! = _bound!(backend, workgroup)
    kernel!(values, cells, cell_neighbours, ndrange = length(values))
    KernelAbstractions.synchronize(backend)
end

@kernel function _bound!(values, cells, cell_neighbours)
    i = @index(Global)

    sum_flux = 0.0
    sum_area = 0
    average = 0.0
    @uniform mzero = eps(eltype(values)) # machine zero

    @inbounds begin
        for fi ∈ cells[i].faces_range
            cID = cell_neighbours[fi]
            sum_flux += max(values[cID], mzero) # bounded sum
            sum_area += 1
        end
        average = sum_flux/sum_area

        values[i] = max(
            max(
                values[i],
                average*signbit(values[i])
            ),
            mzero
        )
    end
end

# Create LUT to map boudnary names to indices
function boundary_map(mesh)
    I = Integer; S = Symbol
    boundary_map = boundary_info{I,S}[]

    for (i, boundary) in enumerate(mesh.boundaries)
        push!(boundary_map, boundary_info{I,S}(i, boundary.name))
    end

    return boundary_map
end