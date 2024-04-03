export StrainRate
export double_inner_product!
export magnitude!, magnitude2!
export number_symbols
# export boundary_map

struct StrainRate{G, GT} <: AbstractTensorField
    gradU::G
    gradUT::GT
end

function (S::StrainRate)(i)
    0.5.*(S.gradU[i] .+ S.gradUT[i])
end

double_inner_product!(
    s, t0::AbstractTensorField, t2) = 
begin
    sum = 0.0
    for i ∈ eachindex(s)
        # t1 = t0[i] .- (1/3)*t0[i]*I
        t1 = 2.0.*t0[i] .- (2/3)*t0[i]*I
        sum = 0.0
        for j ∈ 1:3
            for k ∈ 1:3
                sum +=   t1[j,k]*t2[i][k,j]
                # sum +=   t1[j,k]*t2[i][j,k]
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

function magnitude2!(
    magS::ScalarField, S::AbstractTensorField; scale_factor=1.0
    )
    sum = 0.0
    for i ∈ eachindex(magS.values)
        sum = 0.0
        for j ∈ 1:3
            for k ∈ 1:3
                # sum +=   S[i][j,k]*S[i][k,j]
                sum +=   S[i][j,k]*S[i][j,k]
            end
        end
        magS.values[i] = sum*scale_factor
    end
end

bound!(field, bound) = begin
    mesh = field.mesh
    # (; cells, faces) = mesh
    (; cells, cell_neighbours) = mesh
    for i ∈ eachindex(field)
        sum_flux = 0.0
        sum_area = 0
        average = 0.0
        
        # Cell based average
        # cellsID = cells[i].neighbours
        # for cID ∈ cellsID
        for fi ∈ cells[i].faces_range
            cID = cell_neighbours[fi] # CHECK IF THIS IS CORRECT!!!
            sum_flux += max(field[cID], eps()) # bounded sum
            sum_area += 1
        end
        average = sum_flux/sum_area

        field[i] = max(
            max(
                field[i],
                average*signbit(field[i])
            ),
            bound
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

# function number_symbols(mesh)
#     symbol_mapping = Dict{Symbol, Int}()

#     for (i, boundary) in enumerate(mesh.boundaries)
#         if haskey(symbol_mapping, boundary.name)
#             # Do nothing, the symbol is already mapped
#         else
#             new_number = length(symbol_mapping) + 1
#             symbol_mapping[boundary.name] = new_number
#         end
#     end
    
#     return symbol_mapping
# end