using Plots
using SparseArrays
using LinearOperators
using LinearAlgebra
using IncompleteLU
using Krylov


using FVM_1D.Mesh2D
using FVM_1D.Plotting
using FVM_1D.Discretise
using FVM_1D.Calculate
using FVM_1D.Models
using FVM_1D.Solvers

function generate_mesh()
    n_vertical      = 400 #200 #200
    n_horizontal1   = 500 # 250 #300
    n_horizontal2   = 400 # 200 #400

    # n_vertical      = 20 #400
    # n_horizontal1   = 25 #500
    # n_horizontal2   = 20 #800

    p1 = Point(0.0,0.0,0.0)
    p2 = Point(1.0,0.2,0.0)
    # p2 = Point(1.0,0.0,0.0)
    p3 = Point(1.5,0.2,0.0)
    # p3 = Point(1.5,0.0,0.0)
    p4 = Point(0.0,1.0,0.0)
    p5 = Point(1.0,0.8,0.0)
    # p5 = Point(1.0,1.0,0.0)
    p6 = Point(1.5,0.8,0.0)
    # p6 = Point(1.5,0.7,0.0)
    points = [p1, p2, p3, p4, p5, p6]

    # Edges in x-direction
    e1 = line!(points,1,2,n_horizontal1)
    e2 = line!(points,2,3,n_horizontal2)
    e3 = line!(points,4,5,n_horizontal1)
    e4 = line!(points,5,6,n_horizontal2)

    # Edges in y-direction
    e5 = line!(points,1,4,n_vertical)
    e6 = line!(points,2,5,n_vertical)
    e7 = line!(points,3,6,n_vertical)
    edges = [e1, e2, e3, e4, e5, e6, e7]

    b1 = quad(edges, [1,3,5,6])
    b2 = quad(edges, [2,4,6,7])
    blocks = [b1, b2]

    patch1 = Patch(:inlet,  [5])
    patch2 = Patch(:outlet, [7])
    patch3 = Patch(:bottom, [1,2])
    patch4 = Patch(:top,    [3,4])
    patches = [patch1, patch2, patch3, patch4]

    builder = MeshBuilder2D(points, edges, patches, blocks)
    mesh = generate!(builder)
    return mesh
end

function create_model(::Type{ConvectionDiffusion}, U, J, phi)
    model = ConvectionDiffusion(
        Divergence{Linear}(U, phi),
        Laplacian{Linear}(J, phi),
        0.0
        )
    model.terms.term2.sign[1] = -1
    return model
end

function create_model(::Type{Diffusion}, J, phi)
    model = Diffusion(
        Laplacian{Linear}(J, phi),
        0.0
        )
    return model
end

BCs = (
    Dirichlet(:inlet, 100.0),
    # Dirichlet(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:bottom, 0.0),
    Neumann(:top, 0.0)
    # Dirichlet(:bottom, 50.0),
    # Dirichlet(:top, 50.0)
)

mesh = generate_mesh()
phi = ScalarField(mesh)
equation = Equation(mesh)
phiModel = create_model(ConvectionDiffusion, [4.0, 0.0, 0.0], 1.0, phi)
phiModel = create_model(Diffusion, 1.0, phi)
generate_boundary_conditions!(mesh, phiModel, BCs)

@time discretise!(equation, phiModel, mesh)
@time update_boundaries!(equation, mesh, phiModel, BCs)

res = zeros(length(phi.values))
system = set_solver(equation, BicgstabSolver)
F = ilu(equation.A, τ = 0.005)
opP = LinearOperator(
    Float64, equation.A.m, equation.A.n, false, false, (y, v) -> ldiv!(y, F, v)
)

# mul!(res, op, v, α, β) # α * op * v + β * res

function mymul!(mesh, termIn)
    quote
    function custom!(res, v, α, β::T) where {T}
        (; faces, cells) = $mesh
        term = $termIn
        term_sign = term.sign[1]
        z = zero(eltype(res))
        if β == zero(T)
            @inbounds for cID ∈ eachindex(cells)
                cell = cells[cID]
                (; facesID, nsign, neighbours) = cell
                res[cID] = z
                vC = v[cID]
                @inbounds for fi ∈ eachindex(facesID)
                    fID = facesID[fi]
                    ns = nsign[fi] # normal sign
                    face = faces[fID]
                    nID = neighbours[fi]
                    ap = term_sign*(-term.J * face.area)/face.delta
                    # res[cID] += ap*v[cID]*α
                    res[cID] += ap*vC*α
                    res[cID] += -ap*v[nID]*α
                    # temp = α*ap
                    # vN = v[nID]
                    # res[cID] += (vC - vN)*temp
                end
            end
        else
            @inbounds for cID ∈ eachindex(cells)
                cell = cells[cID]
                (; facesID, nsign, neighbours) = cell
                res[cID] = z
                vC = v[cID]
                @inbounds for fi ∈ eachindex(facesID)
                    fID = facesID[fi]
                    ns = nsign[fi] # normal sign
                    face = faces[fID]
                    nID = neighbours[fi]
                    ap = term_sign*(-term.J * face.area)/face.delta
                    # res[cID] += ap*v[cID]*α
                    res[cID] += ap*vC*α
                    res[cID] += -ap*v[nID]*α
                    # temp = α*ap
                    # vN = v[nID]
                    # res[cID] += (vC - vN)*temp
                end
                res[cID] += β*vC # should it be resC? Seems to work as is!?
            end
        end
    end
    end |> eval
end

myfunc! = mymul!(mesh, phiModel.terms.term1)

A1 = LinearOperator(
    Float64, equation.A.m, equation.A.n, false, false, myfunc!
)

function create_boundary(meshIn, termIn, BCsIn)
    # mul!(res, op, v, α, β) # α * op * v + β * res
quote
    function bc_implicit!(res, v, α, β::T) where {T}
        term = $termIn
        mesh = $meshIn
        BCs = $BCsIn
        term_sign = term.sign[1]
        # term = termIn
        # mesh = meshIn
        (; faces, cells, boundaries) = mesh
        
        if β == zero(T)
            indx = 1 # boundary_index(mesh, :inlet)
            bc = BCs[indx]
            boundary = boundaries[indx]
            (; cellsID, facesID) = boundary
            @inbounds for i ∈ eachindex(cellsID)
                faceID = facesID[i]
                cellID = cellsID[i]
                face = faces[faceID]
                cell = cells[cellID]
                ap = term_sign*(-term.J*face.area/face.delta)
                res[cellID] = α*ap*v[cellID]
            end
            indx = 2 # boundary_index(mesh, :inlet)
            bc = BCs[indx]
            boundary = boundaries[indx]
            (; cellsID, facesID) = boundary
            @inbounds for i ∈ eachindex(cellsID)
                faceID = facesID[i]
                cellID = cellsID[i]
                face = faces[faceID]
                cell = cells[cellID]
                ap = term_sign*(-term.J*face.area/face.delta)
                res[cellID] = α*ap*v[cellID]
            end
        else
            indx = 1 # boundary_index(mesh, :inlet)
            bc = BCs[indx]
            boundary = boundaries[indx]
            (; cellsID, facesID) = boundary
            @inbounds for i ∈ eachindex(cellsID)
                faceID = facesID[i]
                cellID = cellsID[i]
                face = faces[faceID]
                cell = cells[cellID]
                ap = term_sign*(-term.J*face.area/face.delta)
                res[cellID] = α*ap*v[cellID] + β*res[cellID]
            end
            indx = 2 # boundary_index(mesh, :inlet)
            bc = BCs[indx]
            boundary = boundaries[indx]
            (; cellsID, facesID) = boundary
            @inbounds for i ∈ eachindex(cellsID)
                faceID = facesID[i]
                cellID = cellsID[i]
                face = faces[faceID]
                cell = cells[cellID]
                ap = term_sign*(-term.J*face.area/face.delta)
                res[cellID] = α*ap*v[cellID] + β*res[cellID]
            end
        end
    end

    function bc_explicit!(b, term, BCs)
        (; faces, cells, boundaries) = mesh
        term_sign = term.sign[1]
        indx = boundary_index(mesh, :inlet)
        bc = BCs[indx]
        boundary = boundaries[indx]
        (; cellsID, facesID) = boundary
            @inbounds for i ∈ eachindex(cellsID)
                faceID = facesID[i]
                cellID = cellsID[i]
                face = faces[faceID]
                cell = cells[cellID]
                ap = term_sign*(-term.J*face.area/face.delta)
                b[cellID] = ap*bc.value # original
            end
    end
end |> eval
# return f1, f2
end

create_boundary(mesh, phiModel.terms.term1, BCs)

b = zeros(length(mesh.cells))
bc_explicit!(b, phiModel.terms.term1, BCs)
b

A_bc = LinearOperator(
    Float64, equation.A.m, equation.A.n, false, false, bc_implicit!
)


@time Af = A1 + A_bc


system = BicgstabSolver(equation.A, equation.b)
system.x .= 0.0

phi = ScalarField(mesh)
phiModel = create_model(Diffusion, 1.0, phi)
generate_boundary_conditions!(mesh, phiModel, BCs)

phif = FaceScalarField(mesh)
gradf = FaceVectorField(mesh)

gradPhi = Grad{Linear}(phi)
gradPhi = Grad{Linear}(phi,2)

system = set_solver(equation, BicgstabSolver)
discretise!(equation, phiModel, mesh)
update_boundaries!(equation, mesh, phiModel, BCs)
F = ilu(equation.A, τ = 0.005)
# Definition of linear operators to reduce allocations during iterations
m = equation.A.m; n = m
opP = LinearOperator(Float64, m, n, false, false, (y, v) -> ldiv!(y, F, v))
opA = Af
phi.values .= 0.0
# update_residual!(opA, equation, phi1)
bc_explicit!(b, phiModel.terms.term1, BCs)
update_residual!(equation, opA, phi)
system.x .= 0.0
phi.values .= 0.0

@time for i ∈ 1:500
    solve!(system, opA, equation.b, phi.values; M=opP, itmax=100, atol=1e-12, rtol=1e-2)
    relax!(phi, system, 1.0)
    update_residual!(equation, opA, phi)
    if residual(equation) <= 1e-6
        residual_print(equation)
        println("Converged in ", i, " iterations")
        break
    end
end

update_residual!(equation, opA, phi)
mul!(equation.Fx, opA, phi.values)
equation.R .= b .- equation.Fx 
norm(equation.R)
residual(equation)
phi.values .= system.x

plotly(size=(400,400), markersize=1, markerstrokewidth=1)
f(x,y) = 100 + (0.0 - 100)/(1.5 - 0.0)*x
surface(x(mesh), y(mesh), f)
scatter!(x(mesh), y(mesh), phi.values, color=:blue)

Af*phi.values
equation.A*phi.values
A1*phi.values