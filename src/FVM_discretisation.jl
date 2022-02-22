# Discretisation schemes and equation terms (types)
export Linear, Constant, Source, ScalarField, Equation, discretise!,
aP!, aN!, b!, Δ, @defineEqn, @discretise, Mesh, apply_boundary_conditions!, clear!, clearAll!, solve!, generalDiscretise!, sparse_matrix_connectivity, Laplacian, SteadyDiffusion

abstract type AbstractTerm end
abstract type AbstractField end
abstract type AbstractSource end
abstract type AbstractEquation end
struct Linear end
struct Constant end




# struct Equation{I<:Integer,F<:AbstractFloat} <: AbstractEquation
#     A::SparseMatrixCSC{F, I}
#     b::Vector{F}
# end

struct Equation <: AbstractEquation
    A::SparseMatrixCSC{Float64, Int64}
    b::Vector{Float64}
end

struct ScalarField <: AbstractField
    values::Vector{Float64}
    mesh::Mesh
    equation::Equation
end
ScalarField(mesh::Mesh) = begin
    nCells = length(mesh.cells)
    I, J, V = sparse_matrix_connectivity(mesh)
    eqn = Equation(sparse(I,J,V), zeros(nCells))
    ScalarField(zeros(nCells), mesh, eqn)
end

struct Δ{T} <: AbstractTerm end
struct aP!{T} end
struct aN!{T} end
struct b!{T} end

@inline aP!{Linear}(A, term, face, cID) = begin
    A[cID, cID] += (term.Γ * face.area * norm(face.normal)) / face.delta
    nothing
end
@inline  aN!{Linear}(A, term, face, cID, nID) = begin
    A[cID, nID] = -(term.Γ * face.area * norm(face.normal)) / face.delta
    nothing
end
@inline  b!{Linear}(b, term, cID) = begin
    b[cID] = 0.0
    nothing
end
struct Source{T} <: AbstractSource
    ϕ::I where I <: Integer
    type::T
    label::Symbol
end
Source{Constant}(ϕ) = Source{Constant}(ϕ, Constant(), :ConstantSource)

struct Laplacian{T} <: AbstractTerm 
    Γ::Float64
    ϕ::ScalarField
    distretisation::T
end
Laplacian{Linear}(Γ, ϕ) = Laplacian(Γ, ϕ, Linear())

struct SteadyDiffusion{T}
    laplacian::Laplacian{T}
    sign::Vector{Int64}
    equation::Equation
end

# function generalDiscretise!(type, ϕ, J)
function generalDiscretise!(model::SteadyDiffusion{T}, aP!, aN!, b!) where {T}
    laplacian = model.laplacian
    ϕ = laplacian.ϕ
    A = ϕ.equation.A
    b = ϕ.equation.b
    mesh = ϕ.mesh
    cells = mesh.cells
    faces = mesh.faces
    nCells = length(cells)
    for cID ∈ 1:nCells
        cell = cells[cID]
        for fi ∈ eachindex(cell.facesID)
            fID = cell.facesID[fi]
            face = faces[fID]
            nID = cell.neighbours[fi]
            c1 = face.ownerCells[1]
            c2 = face.ownerCells[2]
            aP!(A, laplacian, face, cID)
            if c1 != c2 
            aN!(A, laplacian, face, cID, nID)
            end
        end
        b!(b, laplacian, cID)
    end
    nothing
end

function sparse_matrix_connectivity(mesh::Mesh)
    cells = mesh.cells
    faces = mesh.faces
    nCells = length(cells)
    I = Int[]
    J = Int[]
    for cID = 1:nCells   
        cell = cells[cID]
        for fi ∈ eachindex(cell.facesID)
            fID = cell.facesID[fi]
            face = faces[fID]
            neighbour = cell.neighbours[fi]
            c1 = face.ownerCells[1]
            c2 = face.ownerCells[2]
            I = push!(I, cID)
            J = push!(J, cID)
            # A[cID, cID] += ...
            if c1 != c2
                # A[cID, neighbour] += ...
                I = push!(I, cID)
                J = push!(J, neighbour)
            end
        end
    end
    V = zeros(length(I))
    return I, J, V
end

# Operator overloads
# struct Model{I<:Integer}
#     terms::Vector{AbstractTerm}
#     sources::Vector{AbstractSource}
#     sign::Vector{I}
# end

# Base.:(==)(a::AbstractTerm, b::AbstractSource) = Model(AbstractTerm[a],AbstractSource[b],[1])
# Base.:(==)(a::Vector{AbstractTerm}, b::AbstractSource) = Model(a, [b], [1])
# Base.:(+)(a::TempEquation, b::TempEquation) = begin
#     for i ∈ eachindex(b.A.nzval)
#     b.A.nzval[i] += a.A.nzval[i]
#     end
#     return b
# end
# Base.:(-)(a::AbstractField, b::AbstractField) = begin
#     for i ∈ eachindex(a.equation.A.nzval)
#         a.equation.A.nzval[i] = a.equation.A.nzval[i] - (b.equation.A.nzval[i])
#     end
#     return a
# end

# # Base.:(+)(a::Vector{AbstractTerm}, b::AbstractTerm) = push!(a, b)

# # Macros and functions

# macro defineEqn(eqn)
#     nothing
# end # end macro

# macro discretise(eqn_expr)
# eqn = esc(eqn_expr)
# # terms = :((term1))
# quote
#     terms = [:(term1)]

#     # terms = $(eqn).terms
#     # collect_expr = Symbol[]
#     # for term ∈ terms
#     #     push!(collect_expr, term.arg)
#     # end
#     # args = Expr(:tuple, collect_expr...)
    
#     func = Base.remove_linenums!(:(
#         # function discretise!(eqn::AbstractEquation, $args )
#         function discretise!(eqn::AbstractEquation, term1 )
#         begin
#             mesh = eqn.ϕ.mesh
#             cells = mesh.cells
#             faces = mesh.faces
#             nCells = length(cells)
#             A = eqn.A
#         end
#         for cID ∈ 1:nCells
#             cell = cells[cID]
#             for fi ∈ eachindex(cell.facesID)
#                 fID = cell.facesID[fi]
#                 face = faces[fID]
#                 nID = cell.neighbours[fi]
#                 c1 = face.ownerCells[1]
#                 c2 = face.ownerCells[2]
#                 # discretisation code here
#             end
#         end
#         nothing
#     end))

#     notBoundaryBranch = Base.remove_linenums!(:(if c1 != c2 end))

#     # aP = :(A[cID, cID] += Γ*face.area*norm(face.normal)/face.delta)
#     # aN = :(A[cID, neighbour] += -Γ*face.area*norm(face.normal)/face.delta)

#     for term ∈ terms
#         push!(
#             func.args[2].args[2].args[2].args[2].args[2].args,
#             # term.aP
#             :(aP!(A, $term, face, cID))
#             )

#         push!(
#             notBoundaryBranch.args[2].args,
#             # term.aN
#             :(aN!(A, $term, face, cID, nID))
#             )
#     end
#     push!(
#         func.args[2].args[2].args[2].args[2].args[2].args,
#         notBoundaryBranch)
#     println(func)
#     eval(func)
# end  # end quote
# end

function apply_boundary_conditions!(
    ϕ::ScalarField, k, leftBC, rightBC)
    b = ϕ.equation.b
    mesh = ϕ.mesh
    nCells = length(b)
    faces = mesh.faces
    b[1] = k*faces[1].area*norm(faces[1].normal)/faces[1].delta*leftBC
    b[nCells] = k*faces[nCells+1].area*norm(faces[nCells+1].normal)/faces[nCells+1].delta*rightBC
    nothing
end

function clear!(equation::Equation) 
    equation.A.nzval .= 0.0
    equation.b .= 0.0
    nothing
end

function clear!(ϕ::ScalarField) 
    ϕ.values .= 0.0
    nothing
end

function clearAll!(ϕ::ScalarField)
    ϕ.values .= 0.0
    ϕ.equation.A.nzval .= 0.0
    ϕ.equation.b .= 0.0
    nothing
end

function solve!(ϕ::ScalarField)
    ϕ.values .= ϕ.equation.A\ϕ.equation.b
    nothing
end

#### SOME TESTS WITH KRYLOV.JL

# @time (phi, stats) = cg(ϕEqn.A, ϕEqn.b)
# @time (phi, stats) = bicgstab(ϕEqn.A, ϕEqn.b)

# solverCG = CgSolver(ϕEqn.A, ϕEqn.b)
# @time cg!(
#     solverCG, ϕEqn.A, ϕEqn.b; 
#     M=Krylov.I, atol=1e-12, rtol=1e-4, itmax=1000
#     )
# solverCG.x

# solverBIC = BicgstabSolver(ϕEqn.A, ϕEqn.b)
# @time bicgstab!(
#     solverBIC, ϕEqn.A, ϕEqn.b;
#     M=Krylov.I, N=Krylov.I, atol=1e-12, rtol=1e-4, itmax=1000
#     )

# solverBIC.x
# ϕ.values .= solver.x