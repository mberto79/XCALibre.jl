export AbstractOperators, AbstractLaplacian, AbstractDivergence 
export AbstractScheme 
export AbstractLaplacian, AbstractDivergence 
export Laplacian, Divergence
export Linear, Upwind 
export Discretisation, Equation 

abstract type AbstractOperators end
abstract type AbstractLaplacian <: AbstractOperators end
abstract type AbstractDivergence <: AbstractOperators end
abstract type AbstractScheme end

struct Laplacian{T<:AbstractScheme} <: AbstractLaplacian end
struct Divergence{T<:AbstractScheme} <: AbstractDivergence end

# Supported discretisation schemes
struct Linear <: AbstractScheme end
struct Upwind <: AbstractScheme end

# Types 
struct Discretisation{F,T,I,F1,F2,F3}
    phi::F
    terms::Vector{T}
    signs::Vector{I}
    ap!::F1
    an!::F2 
    b!::F3 
end

struct Equation{I,F}
    A::SparseMatrixCSC{F,I}
    b::Vector{F}
end
Equation(mesh::Mesh2{I,F}) where {I,F} = begin
    nCells = length(mesh.cells)
    i, j, v = sparse_matrix_connectivity(mesh)
    Equation(sparse(i, j, v), zeros(nCells))
end

function sparse_matrix_connectivity(mesh::Mesh2{I,F}) where{I,F}
    cells = mesh.cells
    faces = mesh.faces
    nCells = length(cells)
    i = I[]
    j = I[]
    for cID = 1:nCells   
        cell = cells[cID]
        for fi ∈ eachindex(cell.facesID)
            fID = cell.facesID[fi]
            face = faces[fID]
            neighbour = cell.neighbours[fi]
            # c1 = face.ownerCells[1]
            # c2 = face.ownerCells[2]
            push!(i, cID)
            push!(j, cID)
            # A[cID, cID] += ...
            # if c1 != c2
                # A[cID, neighbour] += ...
                push!(i, cID)
                push!(j, neighbour)
            # end
        end
    end
    v = zeros(F, length(i))
    return i, j, v
end

struct ScalarField{I,F}
    values::Vector{F}
    mesh::Mesh2{I,F}
end

# struct Equation{F}
#     A::SparseCSC
#     b 
# end