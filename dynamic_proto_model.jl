using LinearOperators
using LinearAlgebra

mesh = generate_mesh()

phiBCs = (
        (dirichlet, :inlet, 100),
        (dirichlet, :outlet, 50.0),
        # (neumann, :bottom, 0),
        # (neumann, :top, 0)
        (dirichlet, :bottom, 100),
        (dirichlet, :top, 100)
    )

phi = ScalarField(mesh)
equation = Equation(mesh)

function cell_based1!(res, v, α, β::T; mesh) where T
    (; faces, cells) = mesh
    res .= 0.0
    ap = 0.0
    @inbounds for i ∈ eachindex(res)
        cell = cells[i]
        (; facesID, neighbours, nsign) = cell
        @inbounds for fi ∈ eachindex(facesID)
            face = faces[facesID[fi]]
            (; normal, delta, area) = face
            ns = nsign[fi]
            ap_temp =  (-1.0 * area)/delta # J = 1.0
            ap += ap_temp 
            an = -ap_temp
            res[i] += an*v[neighbours[fi]]
        end
        res[i] += ap*v[i]
        ap = 0.0
    end
end
cell_based!(res, v, α, β::T) where T = begin
    cell_based1!(res, v, 1.0, 0.0; mesh=mesh)
end

# face-based calculation
function face_based1!(res, v, α, β::T; mesh) where T
    (; faces, cells) = mesh
    start = total_boundary_faces(mesh) + 1
    finish = length(faces)
    @inbounds @simd for i ∈  eachindex(res)
        res[i] = 0.0
    end
    @inbounds for fID ∈ start:finish
        face = faces[fID]
        (; ownerCells, normal, delta, area) = face
        cID1 = ownerCells[1]
        cID2 = ownerCells[2]
        ap =  -1.0*area/delta
        an = -ap
        res[cID1] += ap*v[cID1]
        res[cID1] += an*v[cID2]
        res[cID2] += ap*v[cID2]
        res[cID2] += an*v[cID1]
    end
end
face_based!(res, v, α, β::T) where T = begin
    face_based1!(res, v, 1.0, 0.0; mesh=mesh)
end


function flux!(J, mesh) where T
    (; faces, cells) = mesh
    start = total_boundary_faces(mesh) + 1
    finish = length(faces)
    @inbounds @simd for i ∈  eachindex(J)
        J[i] = 0.0
    end
    @inbounds for fID ∈ start:finish
        face = faces[fID]
        (; delta, area) = face
        J[fID] =  -1.0*area/delta
    end
end

function fluxC!(JC, equation) where T
    (; A) = equation
    @inbounds @simd for i ∈  eachindex(JC)
        JC[i] = 0.0
    end
    @inbounds for i ∈ eachindex(equation.b)
        JC[i] =  A[i,i]
    end
end

# face-based calculation (pre-allocated)
function face_flux1!(res, v, α, β::T; mesh, JF) where T
    (; faces, cells) = mesh
    start = total_boundary_faces(mesh) + 1
    finish = length(faces)
    @inbounds @simd for i ∈  eachindex(res)
        res[i] = 0.0
    end
    @inbounds for fID ∈ start:finish
        face = faces[fID]
        (; ownerCells) = face
        cID1 = ownerCells[1]
        cID2 = ownerCells[2]
        ap =  JF[fID]
        an = -ap
        res[cID1] = 2.5 + fID
        # res[cID1] += ap*v[cID1]
        # res[cID1] += an*v[cID2]
        # res[cID2] += ap*v[cID2]
        # res[cID2] += an*v[cID1]
    end
end
face_flux!(res, v, α, β::T) where T = begin
    face_flux1!(res, v, 1.0, 0.0; mesh=mesh, JF=JF)
end

# face-based calculation (pre-allocated)
function face_fluxC1!(res, v, α, β::T; mesh, JF, JC) where T
    (; faces, cells) = mesh
    start = total_boundary_faces(mesh) + 1
    finish = length(faces)
    @inbounds @simd for i ∈  eachindex(res)
        res[i] = 0.0
    end
    @inbounds for fID ∈ start:finish
        face = faces[fID]
        (; ownerCells, normal, delta, area) = face
        cID1 = ownerCells[1]
        cID2 = ownerCells[2]
        ac = JC[fID]
        ap =  JF[fID]
        an = -ap
        res[cID1] = ac*v[cID1]
        res[cID1] += an*v[cID2]
        res[cID2] = ac*v[cID2]
        res[cID2] += an*v[cID1]
    end
end
face_fluxC!(res, v, α, β::T) where T = begin
    face_fluxC1!(res, v, 1.0, 0.0; mesh=mesh, JF=JF, JC=JC)
end

JF = zeros(length(mesh.faces))
flux!(JF, mesh)
JF

JC = zeros(length(mesh.cells))
fluxC!(JC, equation)
JC

ncells = length(mesh.cells)
opA = LinearOperator(
    Float64, ncells, ncells, false, false, 
    cell_based!, nothing, nothing
    )

opAf = LinearOperator(
 Float64, length(mesh.cells), length(mesh.cells), false, false,
    face_based!, nothing, nothing
    )

opAflux = LinearOperator(
    Float64, length(mesh.cells), length(mesh.cells), false, false,
    face_flux!, nothing, nothing
)

opAfluxC = LinearOperator(
    Float64, length(mesh.cells), length(mesh.cells), false, false,
    face_fluxC!, nothing, nothing
)

has_args5(opA)

@time opAm = LinearOperator(equation.A)

@time At = transpose(equation.A)

vector = 2.0.*rand(length(mesh.cells))
out1 = zeros(length(mesh.cells))
out2 = zeros(length(mesh.cells))
out3 = zeros(length(mesh.cells))
out4 = zeros(length(mesh.cells))
@time mul!(out1, opAm, vector)
@time mul!(out2, opAf, vector)
@time mul!(out3, opAflux, vector)
@time mul!(out4, equation.A, vector)


test(out3, opAA, vector)

out1./out2
out1./out3
out2./out3
out4

phi = ScalarField(mesh)
equation = Equation(mesh)

J = 1.0
phiModel = SteadyDiffusion(Laplacian{Linear}(J, phi), 0.0)
phiModel.terms.term1.sign[1] = 1
generate_boundary_conditions!(mesh, phiModel, phiBCs)
@time discretise4!(equation, phiModel, mesh)
update_boundaries!(equation, mesh, phiModel, phiBCs)
phi.values .= equation.A\equation.b

equation.A.nzval .= 0.0
opBCs = LinearOperator(equation.A)
opAA = opAf + opBCs

Matrix(opBCs)
Matrix(opAf)
Matrix(opAA)

using FVM_1D.Solvers
using Krylov
using IncompleteLU
using LinearOperators
using LinearAlgebra

# GmresSolver(opAA, equation.b)
# @time F = ilu(opAA, τ = 0.005)
# n = length(b)
# bl = false
# opP = LinearOperator(Float64, n, n, bl, bl, (y, v) -> ldiv!(y, F, v))

atol=1e-8
rtol=1e-7
itmax=500

phi.values .= 100.0
(; b, R, Fx) = equation
mul!(Fx, opAA, phi.values)
R .= b .- Fx
mul!(Fx, equation.A, phi.values)
R .= b .- Fx
system = GmresSolver(opAA, equation.b)
system = BicgstabSolver(opAA, equation.b)
system = GmresSolver(equation.A, equation.b)
system = BicgstabSolver(equation.A, equation.b)

@time solve!(system, opAA, R; itmax=itmax, atol=atol, rtol=rtol)
@time solve!(system, equation.A, R; itmax=itmax, atol=atol, rtol=rtol)
phi.values .+= solution(system)

@time phi.values .= equation.A\equation.b
println("Residual: ", norm(R))
phi.values .= 100.0
phi.values