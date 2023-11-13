using CUDA
using StaticArrays
using Adapt

# CUDA.allowscalar(true)
# CUDA.allowscalar(false)

struct Test0{T,F}
    centre::F
    nodesID::T
end
Adapt.@adapt_structure Test0


# Humberto's code - push onto array on CPU, nodesID vector defined as CPU vector type
container = Test0{Vector{Int64}, Float64}[]
for i in 1:5
    # push!(container, Test0(rand(), rand(2:50, 3)))
    push!(container, Test0(rand(), rand(2:50, rand(2:20))))
end

container_gpu = cu.(container)
solution = similar(container_gpu)

container_gpu

# Array definitions - constructors likely needed in type definition files
nodes = typeof(container_gpu[1].nodesID)[]
centres = typeof(container_gpu[1].centre)[]

for i in 1:length(container_gpu)
    push!(nodes,container_gpu[i].nodesID)
    push!(centres,container_gpu[i].centre)
end

nodes
centres = cu(centres)

function test_kernel!(n,c)
    i = threadIdx().x

    @inbounds if i <= length(c) && i>0

        c[i] = c[i] + c[i]

    end

    @inbounds if i <= length(n) && i>0

        n[i] = n[i] + n[i]
        
    end

    return nothing
end

for i in 1:length(nodes)
@cuda threads = length(max(length(nodes[i]),length(centres))) test_kernel!(nodes[i],centres)
end
centres
nodes