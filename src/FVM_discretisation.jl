# Discretisation schemes and equation terms (types)
export Linear, Constant, Source, ScalarField, Equation, discretise!,
aP!, aN!, b!,
Δ, @defineEqn, @discretise, Mesh, apply_boundary_conditions!, clear!, solve!

struct ScalarField{I,F}
    values::Vector{F}
    mesh::Mesh{I,F}
end
# ScalarField(mesh::Mesh) = begin
#     nothing
# end

abstract type AbstractTerm end
abstract type AbstractSource end
abstract type AbstractEquation end
struct Linear end
struct Constant end

struct Δ{T,I,F} <: AbstractTerm 
    # Γ::Union{Float32,Float64,Int32,Int64}
    Γ::F 
    ϕ::ScalarField{I,F}
    distretisation::T
    label::Symbol
end
function Δ{Linear}(Γ, ϕ::ScalarField) 
    Δ(Γ, ϕ, Linear(), :Laplacian)
end
@inline aP!(A, term::Δ{Linear}, face, cID) = begin
    A[cID, cID] += (term.Γ * face.area * norm(face.normal)) / face.delta
    nothing
end
@inline  aN!(A, term::Δ{Linear}, face, cID, nID) = begin
    A[cID, nID] += -(term.Γ * face.area * norm(face.normal)) / face.delta
    nothing
end
@inline  b!(b, term::Δ{Linear}, face, cID) = begin
    b[cID] += 0.0
    nothing
end

struct Source{T} <: AbstractSource
    ϕ::I where I <: Integer
    type::T
    label::Symbol
end
Source{Constant}(ϕ) = Source{Constant}(ϕ, Constant(), :ConstantSource)


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

struct Equation{I,F} <: AbstractEquation
    ϕ::ScalarField{I,F}
    # A::Union{Matrix{F},SparseMatrixCSC{F, I}}
    A::SparseMatrixCSC{F, I}
    b::Vector{F}
end
function Equation(ϕ) 
    ncells = length(ϕ.mesh.cells)
    I, J, V = sparse_matrix_connectivity(ϕ.mesh)
    Equation(ϕ, sparse(I,J,V), zeros(ncells))
end

# Operator overloads
struct Model{I<:Integer}
    terms::Vector{AbstractTerm}
    sources::Vector{AbstractSource}
    sign::Vector{I}
end
Base.:(==)(a::AbstractTerm, b::AbstractSource) = Model(AbstractTerm[a],AbstractSource[b],[1])
Base.:(==)(a::Vector{AbstractTerm}, b::AbstractSource) = Model(a, [b], [1])
Base.:(+)(a::AbstractTerm, b::AbstractTerm) = AbstractTerm[a,b]
Base.:(+)(a::Vector{AbstractTerm}, b::AbstractTerm) = push!(a, b)

# Macros and functions

macro defineEqn(eqn)
    nothing
end # end macro

macro discretise(eqn_expr)
eqn = esc(eqn_expr)
# terms = :((term1))
quote
    terms = [:(term1)]

    # terms = $(eqn).terms
    # collect_expr = Symbol[]
    # for term ∈ terms
    #     push!(collect_expr, term.arg)
    # end
    # args = Expr(:tuple, collect_expr...)
    
    func = Base.remove_linenums!(:(
        # function discretise!(eqn::AbstractEquation, $args )
        function discretise!(eqn::AbstractEquation, term1 )
        begin
            mesh = eqn.ϕ.mesh
            cells = mesh.cells
            faces = mesh.faces
            nCells = length(cells)
            A = eqn.A
        end
        for cID ∈ 1:nCells
            cell = cells[cID]
            for fi ∈ eachindex(cell.facesID)
                fID = cell.facesID[fi]
                face = faces[fID]
                nID = cell.neighbours[fi]
                c1 = face.ownerCells[1]
                c2 = face.ownerCells[2]
                # discretisation code here
            end
        end
        nothing
    end))

    notBoundaryBranch = Base.remove_linenums!(:(if c1 != c2 end))

    # aP = :(A[cID, cID] += Γ*face.area*norm(face.normal)/face.delta)
    # aN = :(A[cID, neighbour] += -Γ*face.area*norm(face.normal)/face.delta)

    for term ∈ terms
        push!(
            func.args[2].args[2].args[2].args[2].args[2].args,
            # term.aP
            :(aP!(A, $term, face, cID))
            )

        push!(
            notBoundaryBranch.args[2].args,
            # term.aN
            :(aN!(A, $term, face, cID, nID))
            )
    end
    push!(
        func.args[2].args[2].args[2].args[2].args[2].args,
        notBoundaryBranch)
    println(func)
    eval(func)
end  # end quote
end

function apply_boundary_conditions!(
    eqn::AbstractEquation, k, leftBC, rightBC) # where {I <: Integer, F}
    b = eqn.b
    mesh = eqn.ϕ.mesh
    nCells = length(b)
    faces = mesh.faces
    b[1] = k*faces[1].area*norm(faces[1].normal)/faces[1].delta*leftBC
    b[nCells] = k*faces[nCells+1].area*norm(faces[nCells+1].normal)/faces[nCells+1].delta*rightBC
    nothing
end

function clear!(eqn::AbstractEquation) # where {I,F}
    eqn.A.nzval .= 0.0
    eqn.b .= 0.0
    nothing
end

function solve!(eqn::AbstractEquation)
    eqn.ϕ.values .= eqn.A\eqn.b
    return eqn
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