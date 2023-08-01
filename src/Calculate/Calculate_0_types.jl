export Grad, Div
export get_scheme

# Gradient explicit operator

struct Grad{S<:AbstractScheme,F,X,Y,Z,I,M}
    phi::F
    x::X
    y::Y
    z::Z
    # phif::FaceScalarField{I,F} 
    correctors::I
    correct::Bool
    mesh::M
end
Grad{S}(phi::ScalarField) where S= begin
    mesh = phi.mesh
    (; cells) = mesh
    ncells = length(cells)
    ell = eltype(mesh.nodes[1].coords)
    gradx = zeros(ell, ncells)
    grady = zeros(ell, ncells)
    gradz = zeros(ell, ncells)
    F = typeof(phi)
    X = Y = Z = typeof(gradx)
    I = eltype(mesh.nodes[1].neighbourCells)
    M = typeof(mesh)
    Grad{S,F,X,Y,Z,I,M}(phi, gradx, grady, gradz, one(I), false, mesh)
end
Grad{S}(vec::VectorField) where S = begin
    println("Definition for vector")
end

Grad{S}(phi::ScalarField, correctors::Integer) where S = begin 
    mesh = phi.mesh
    (; cells) = mesh
    ncells = length(cells)
    F = eltype(mesh.nodes[1].coords)
    I = eltype(mesh.nodes[1].neighbourCells)
    SF = typeof(phi)
    M = typeof(mesh)
    gradx = zeros(F, ncells)
    grady = zeros(F, ncells)
    gradz = zeros(F, ncells)
    Grad{S,I,F,SF,M}(phi, gradx, grady, gradz, correctors, true, mesh)
end
get_scheme(term::Grad{S,I,F}) where {S,I,F} = S
# (grad::Grad{S,I,F})(i::I) where {S,I,F} = SVector{3,F}(grad.x[i], grad.y[i], grad.z[i])
Base.getindex(grad::Grad{S,I,F}, i::Integer) where {S,I,F} = SVector{3,F}(grad.x[i], grad.y[i], grad.z[i])

# Divergence explicit operator

struct Div{VF<:VectorField,FVF<:FaceVectorField,F,M}
    vector::VF
    face_vector::FVF
    values::Vector{F}
    mesh::M
end
Div(vector::VectorField) = begin
    mesh = vector.mesh
    face_vector = FaceVectorField(mesh)
    values = zeros(F, length(mesh.cells))
    Div(vector, face_vector, values, mesh)
end