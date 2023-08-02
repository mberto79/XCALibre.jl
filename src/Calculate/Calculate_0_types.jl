export Grad, Div, Transpose
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
Grad{S}(psi::VectorField) where S = begin
    mesh = psi.mesh
    gradx = Grad{S}(psi.x)
    grady = Grad{S}(psi.y)
    gradz = Grad{S}(psi.z)
    F = typeof(psi)
    X = typeof(gradx)
    Y = typeof(gradx)
    Z = typeof(gradx)
    I = eltype(mesh.nodes[1].neighbourCells)
    M = typeof(mesh)
    Grad{S,F,X,Y,Z,I,M}(psi, gradx, grady, gradz, one(I), false, mesh)
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
Base.getindex(grad::Grad{S,F,X,Y,Z,I,M}, i::Integer) where {S,F,X<:AbstractVector,Y,Z,I,M} = begin
    SVector{3,F}(grad.x[i], grad.y[i], grad.z[i])
end

Base.getindex(grad::Grad{S,F,X,Y,Z,I,M}, i::Integer) where {S,F,X<:Grad,Y,Z,I,M} = begin
    Tf = eltype(grad.x.x)
    SMatrix{3,3,Tf,9}(
        grad.x.x[i],
        grad.x.y[i],
        grad.x.z[i],
        grad.y.x[i],
        grad.y.y[i],
        grad.y.z[i],
        grad.z.x[i],
        grad.z.y[i],
        grad.z.z[i],
        )
end

struct Transpose{T<:Grad}
    parent::T
end
Base.getindex(t::Transpose{Grad{S,F,X,Y,Z,I,M}}, i::Integer) where {S,F,X<:Grad,Y,Z,I,M} = begin
    gradt = t.parent
    Tf = eltype(gradt.x.x)
    SMatrix{3,3,Tf,9}(
        gradt.x.x[i],
        gradt.y.x[i],
        gradt.z.x[i],
        gradt.x.y[i],
        gradt.y.y[i],
        gradt.z.y[i],
        gradt.x.z[i],
        gradt.y.z[i],
        gradt.z.z[i],
        )
end

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

