export Grad
export grad!, source!

# Define Gradient type and functionality

struct Grad{S<:AbstractScheme,F,X,Y,Z,I,M}
    phi::F
    x::X
    y::Y
    z::Z
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

# GRADIENT CALCULATION FUNCTIONS

function source!(grad::Grad{S,I,F}, phif, phi, BCs; source=true) where {S,I,F}
    grad!(grad, phif, phi, BCs; source=source)
end

# Mid-point gradient calculation

function grad!(grad::Grad{Midpoint,TI,TF}, phif, phi, BCs; source=false) where {TI,TF}
    interpolate!(get_scheme(grad), phif, phi)
    correct_boundaries!(phif, phi, BCs)
    green_gauss!(grad, phif; source)
    for i ∈ 1:2
        correct_interpolation!(grad, phif, phi)
        green_gauss!(grad, phif; source)
    end
end

function correct_interpolation!(
    grad::Grad{Midpoint,TI,TF}, phif::FaceScalarField{TI,TF}, phi::ScalarField{TI,TF}
    ) where {TI,TF}
    (; mesh, values) = phif
    (; faces, cells) = mesh
    phic = phi.values
    nbfaces = total_boundary_faces(mesh)
    start = nbfaces + 1
    @inbounds @simd for fID ∈ start:length(faces)
        face = faces[fID]
        ownerCells = face.ownerCells
        owner1 = ownerCells[1]
        owner2 = ownerCells[2]
        cell1 = cells[owner1]
        cell2 = cells[owner2]
        phi1 = phic[owner1]
        phi2 = phic[owner2]
        ∇phi1 = grad(owner1)
        ∇phi2 = grad(owner2)
        weight = 0.5
        rf = face.centre 
        rP = cell1.centre 
        rN = cell2.centre
        phifᵖ = weight*(phi1 + phi2)
        ∇phi = weight*(∇phi1 + ∇phi2)
        R = rf - weight*(rP + rN)
        values[fID] = phifᵖ + ∇phi⋅R
    end
end

# Linear gradient calculation

function grad!(grad::Grad{Linear,TI,TF}, phif, phi, BCs; source=false) where {TI,TF}
    interpolate!(phif, phi)
    correct_boundaries!(phif, phi, BCs)
    green_gauss!(grad, phif; source)
    # for i ∈ 1:2
    #     correct_interpolation!(grad, phif, phi)
    #     green_gauss!(grad, phif; source)
    # end
end