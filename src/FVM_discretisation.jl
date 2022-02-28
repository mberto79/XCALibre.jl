# Discretisation schemes and equation terms (types)
export @discretise
export ScalarField, Equation
export Laplacian, Divergence, Source
export Linear, Constant
export aP!, aN!, b!
export apply_boundary_conditions!, clear!, clearAll!, solve!

# # Macros and functions

macro discretise(Model_type, nTerms::Integer, nSources::Integer)
    aP! = Expr(:block)
    aN! = Expr(:block)
    b!  = Expr(:block)
    for t ∈ 1:nTerms
        push!(aP!.args, :(aP!(model.terms.$(Symbol("term$t")), A, face, cID)))
        push!(aN!.args, :(aN!(model.terms.$(Symbol("term$t")), A, face, cID, nID)))
        push!(b!.args, :(b!(model.terms.$(Symbol("term$t")), b, cID)))
    end 
    
    quote 
        function discretise!(model::$Model_type, equation)
            mesh = model.terms.term1.ϕ.mesh
            cells = mesh.cells
            faces = mesh.faces
            A = equation.A
            b = equation.b
            @inbounds for cID ∈ eachindex(cells)
                cell = cells[cID]
                A[cID,cID] = zero(0.0)
                @inbounds for fi ∈ eachindex(cell.facesID)
                    fID = cell.facesID[fi]
                    face = faces[fID]
                    nID = cell.neighbours[fi]
                    c1 = face.ownerCells[1]
                    c2 = face.ownerCells[2]
                    $aP!
                    if c1 != c2 
                        A[cID,nID] = zero(0.0)
                        $aN!                    
                    end
                end
                b[cID] = zero(0.0)
                $b!
            end
            nothing
        end # end function
    end |> esc # end quote and escape!
end # end macro


abstract type AbstractTerm end
abstract type AbstractField end
abstract type AbstractSource end
abstract type AbstractEquation end
struct Linear end
struct Constant end

struct Equation <: AbstractEquation
    A::SparseMatrixCSC{Float64, Int64}
    b::Vector{Float64}
end
Equation(mesh::Mesh) = begin
    nCells = length(mesh.cells)
    I, J, V = sparse_matrix_connectivity(mesh)
    Equation(sparse(I,J,V), zeros(nCells))
end

struct ScalarField <: AbstractField
    values::Vector{Float64}
    mesh::Mesh
    equation::Equation
end
ScalarField(mesh::Mesh, equation::Equation) = begin
    nCells = length(mesh.cells)
    ScalarField(zeros(nCells), mesh, equation)
end

### OPERATORS AND SCHEMES
struct Source{T} <: AbstractSource
    ϕ::Float64
    type::T
    label::Symbol
end
Source{Constant}(ϕ) = Source{Constant}(ϕ, Constant(), :ConstantSource)

struct Laplacian{T} <: AbstractTerm 
    J::Float64
    ϕ::ScalarField
    sign::Vector{Int64}
end
Laplacian{Linear}(J, ϕ) = Laplacian{Linear}(J, ϕ, [1])
@inline aP!(term::Laplacian{Linear}, A, face, cID) = begin
    A[cID, cID] += term.sign[1]*(term.J * face.area * norm(face.normal))/face.delta
    nothing
end
@inline aN!(term::Laplacian{Linear}, A, face, cID, nID) = begin
    A[cID, nID] += term.sign[1]*(-term.J * face.area * norm(face.normal))/face.delta
    nothing
end
@inline b!(term::Laplacian{Linear}, b, cID) = begin
    b[cID] = 0.0
    nothing
end

struct Divergence{T} <: AbstractTerm 
    J::SVector{3, Float64}
    ϕ::ScalarField
    sign::Vector{Int64}
end
Divergence{Linear}(J, ϕ) = Divergence{Linear}(J, ϕ, [1])
@inline aP!(term::Divergence{Linear}, A, face, cID) = begin
    A[cID, cID] += term.sign[1]*(term.J ⋅ face.normal * face.area)/2.0
    nothing
end
@inline aN!(term::Divergence{Linear}, A, face, cID, nID) = begin
    A[cID, nID] += term.sign[1]*(term.J ⋅ face.normal * face.area)/2.0
    nothing
end
@inline b!(term::Divergence{Linear}, b, cID) = begin
    b[cID] = 0.0
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

function apply_boundary_conditions!(
    equation::Equation, mesh::Mesh, k, U, leftBC, rightBC)
    A = equation.A
    b = equation.b
    nCells = length(b)
    faces = mesh.faces
    b[1] = -k*faces[1].area*norm(faces[1].normal)/faces[1].delta*leftBC
    b[nCells] = -k*faces[nCells+1].area*norm(faces[nCells+1].normal)/faces[nCells+1].delta*rightBC
    b[1] += (U ⋅ faces[1].normal * faces[1].area * leftBC)
    # A[1,1] -= (U ⋅ faces[1].normal * faces[1].area)/2.0
    b[nCells] += (U ⋅ faces[nCells+1].normal * faces[nCells+1].area * rightBC)
    # A[nCells,nCells] -= (U ⋅ faces[nCells+1].normal * faces[nCells+1].area)/2.0

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