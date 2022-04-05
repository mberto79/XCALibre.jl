using LinearOperators
using LinearAlgebra

iter_A(x) = [ -2x[1] + x[2]; x[1:end-2] - 2x[2:end-1] + x[3:end] ; -2x[end] + x[end-1] ]

function customfunc1!(res, v, α, β::T; mesh) where T
    (; faces, cells) = mesh
    ap = 0.0
    @inbounds for i ∈ eachindex(res)
        cell = cells[i]
        (; facesID, neighbours, nsign) = cell
        @inbounds for fi ∈ eachindex(facesID)
            face = faces[facesID[fi]]
            (; normal, delta, area) = face
            ns = nsign[fi]
            ap += (-1.0 * area)/delta # J = 1.0
            an = -ap
            res[i] += an*v[neighbours[fi]]
        end
        res[i] += ap*v[i]
        ap = 0.0
    end
end
customfunc2!(res, v, α, β::T) where T = begin
    customfunc1!(res, v, 1.0, 0.0; mesh=mesh)
end
function tcustomfunc!(res, w, α, β::T) where T
    if β == zero(T)
        res[1] = w[1] * α
        res[2] =  (w[1] + w[2]) * α
    else
        res[1] = w[1] * α + res[1] * β
        res[2] =  (w[1] + w[2]) * α + res[2] * β
    end
    nothing
end

ncells = length(mesh.cells)
opA = LinearOperator(
    Float64, ncells, ncells, false, false, 
    customfunc2!, nothing, nothing
    )

has_args5(opA)

opAm = LinearOperator(equation.A)

vector = ones(ncells)
out1 = zeros(ncells)
out2 = zeros(ncells)
out3 = zeros(ncells)
@time mul!(out1, opA, vector)
@time mul!(out2, equation.A, vector)
@time mul!(out3, opAm, vector)
d = [equation.A[i,i] for i ∈ 1:ncells]

compare = [(d[i], test[i]) for i ∈ 1:ncells]

equation.A[1,1]
