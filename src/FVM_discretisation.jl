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
        push!(aP!.args, :(aP!(
            model.terms.$(Symbol("term$t")), A, cell, face, nsign, cID)
            ))
        push!(aN!.args, :(aN!(
            model.terms.$(Symbol("term$t")), A, cell, face, nsign, cID, nID)
            ))
        push!(b!.args, :(
            b!(model.terms.$(Symbol("term$t")), b, cell, cID)
            ))
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
                    nsign = cell.nsign[fi]
                    face = faces[fID]
                    nID = cell.neighbours[fi]
                    c1 = face.ownerCells[1]
                    c2 = face.ownerCells[2]
                    if c1 != c2 
                        A[cID,nID] = zero(0.0)
                        $aP!
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
@inline aP!(term::Laplacian{Linear}, A, cell, face, nsign, cID) = begin
    A[cID, cID] += term.sign[1]*(-term.J * face.area)/face.delta
    nothing
end
@inline aN!(term::Laplacian{Linear}, A, cell, face, nsign, cID, nID) = begin
    A[cID, nID] += term.sign[1]*(term.J * face.area)/face.delta
    nothing
end
@inline b!(term::Laplacian{Linear}, b, cell, cID) = begin
    b[cID] = 0.0
    nothing
end

struct Divergence{T} <: AbstractTerm 
    J::SVector{3, Float64}
    ϕ::ScalarField
    sign::Vector{Int64}
end
Divergence{Linear}(J, ϕ) = Divergence{Linear}(J, ϕ, [1])
@inline aP!(term::Divergence{Linear}, A, cell, face, nsign, cID) = begin
    A[cID, cID] += term.sign[1]*(term.J⋅face.normal*nsign*face.area)/2.0
    nothing
end
@inline aN!(term::Divergence{Linear}, A, cell, face, nsign, cID, nID) = begin
    A[cID, nID] += term.sign[1]*(term.J⋅face.normal*nsign*face.area)/2.0
    nothing
end
@inline b!(term::Divergence{Linear}, b, cell, cID) = begin
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

function face_properties(mesh::Mesh, facei::Integer)
    faces = mesh.faces
    area = faces[facei].area
    delta = faces[facei].delta
    normal = faces[facei].normal
    nsign = mesh.cells[faces[facei].ownerCells[1]].nsign[1]
    return area, delta, normal, nsign
end

function apply_boundary_conditions!(
    equation::Equation, mesh::Mesh, model, leftBC, rightBC)
    # equation::Equation, mesh::Mesh, k, U, leftBC, rightBC)
    A = equation.A
    b = equation.b
    nCells = length(b)
    begin
        J = model.terms.term1.J
        sgn = model.terms.term1.sign[1]

        area, delta, normal, nsign = face_properties(mesh, 1)
        b[1] += sgn*(J⋅(area*normal*nsign))*leftBC

        area, delta, normal, nsign = face_properties(mesh, nCells+1)
        b[nCells] += sgn*(J⋅(area*normal*nsign))*rightBC
    end
    
    begin
        J = model.terms.term2.J
        sgn = model.terms.term2.sign[1]

        area, delta, normal, nsign = face_properties(mesh, 1)
        b[1] += sgn*(-J*area/delta*leftBC)
        A[1,1] += sgn*(-J*area/delta)

        area, delta, normal, nsign = face_properties(mesh, nCells+1)
        b[nCells] += sgn*(-J*area/delta*rightBC)
        A[nCells,nCells] += sgn*(-J*area/delta)

    end
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