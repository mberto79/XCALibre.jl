using Plots

using FVM_1D

using Krylov

# backwardFacingStep_2mm, backwardFacingStep_10mm
mesh_file = "unv_sample_meshes/backwardFacingStep_10mm.unv"
mesh = build_mesh(mesh_file, scale=0.001)

p = ScalarField(mesh)
U = VectorField(mesh)


function (v::VectorField)(s::Symbol)
    getproperty(v, s)
end

function test(v, s)
     v(s)
     nothing
end

@time test(U,:x)

velocity = [0.5, 0.0, 0.0]
nu = 1e-3
Re = velocity[1]*0.1/nu

U = assign(
    U,
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Dirichlet(:wall, [0.0, 0.0, 0.0]),
    Dirichlet(:top, [0.0, 0.0, 0.0])
    # Neumann(:top, 0.0)
    )

p = assign(
    p,
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:wall, 0.0),
    Neumann(:top, 0.0)
)

setup_U = SolverSetup(
    solver      = GmresSolver, # BicgstabSolver, GmresSolver
    relax       = 0.8,
    itmax       = 100,
    rtol        = 1e-1
)

setup_p = SolverSetup(
    solver      = GmresSolver, # GmresSolver, FomSolver, DiomSolver
    relax       = 0.3,
    itmax       = 100,
    rtol        = 1e-1
)

GC.gc()

initialise!(U, velocity)
initialise!(p, 0.0)

gradU = Grad{Linear}(U)
gradUT = Transpose(gradU)
Uf = FaceVectorField(mesh)
interpolate!(Uf, U)   
correct_boundaries!(Uf, U, U.BCs)

test(gradU, Uf, U) = begin
    source!(gradU.x, Uf.x, U.x, U.x.BCs)
    source!(gradU.y, Uf.y, U.y, U.y.BCs)
    source!(gradU.z, Uf.z, U.z, U.z.BCs)
end

gradU[20]
gradUT[20]
using LinearAlgebra
@time test(gradU, Uf, U)
S = zeros(Float64, length(mesh.cells))
for i ∈ eachindex(S)
    S[i] = norm(gradU[i])
end

p.values .= S
write_vtk("results", mesh, ("U", U), ("p", p))

struct Transpose{T}
    parent::T
end


x(mesh)
scatter(x(mesh), y(mesh), U.x.values)
scatter(x(mesh), y(mesh), gradU.x.x)
surface(x(mesh), y(mesh), S, viewangle=(0,90))
iterations = 1000
Rx, Ry, Rp = isimple!( # 123 its, 4.68k allocs
    mesh, nu, U, p,
    # setup_U, setup_p, iterations, pref=0.0)
    setup_U, setup_p, iterations)

write_vtk("results", mesh, ("U", U), ("p", p))

plot(; xlims=(0,123))
plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
plot!(1:length(Rp), Rp, yscale=:log10, label="p")