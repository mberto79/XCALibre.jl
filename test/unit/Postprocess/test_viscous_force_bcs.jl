using XCALibre
using StaticArrays
using LinearAlgebra
# viscous_force and wall_shear_stress integrate the boundary viscous stress of the momentum equation,
# taking the wall velocity of each face from the velocity condition of the patch: a fixed vector
# (Wall, Dirichlet), a user function at the face centre and `time`, or a rotation velocity.

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "quad40.unv"), scale=0.001)
backend = CPU()
hardware = Hardware(backend=backend, workgroup=1024)
mesh_dev = adapt(backend, mesh)

model = Physics(
    time = Transient(),
    fluid = Fluid{Incompressible}(nu=1e-3),
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev)

# a non-uniform velocity field, so every face sees a different velocity difference
(; U) = model.momentum
for i ∈ eachindex(U.x.values)
    c = mesh_dev.cells.centre[i]
    x, y = c[1], c[2]
    U.x.values[i] = 0.4 + 30x + 200y
    U.y.values[i] = -0.1 + 50x*y*1e3
end

pBCs = [Zerogradient(:inlet), Dirichlet(:outlet, 0.0), Zerogradient(:bottom), Zerogradient(:top)]
vf_config = bottom -> Configuration(
    solvers=(;), schemes=(;), runtime=Runtime(iterations=1, time_step=1, write_interval=-1),
    hardware=hardware, postprocess=nothing,
    boundaries=assign(region=mesh_dev, (
        U=[Zerogradient(:inlet), Zerogradient(:outlet), bottom, Zerogradient(:top)], p=pBCs)))

Uw = [0.3, -0.05, 0.0]
config_wall = vf_config(Wall(:bottom, Uw))
config_wall2 = vf_config(Wall(:bottom, 2Uw))
config_dir = vf_config(Dirichlet(:bottom, Uw))
config_fun = vf_config(DirichletFunction(:bottom, (c, t, i) -> SVector(0.3, -0.05, 0.0)))
config_funt = vf_config(DirichletFunction(:bottom, (c, t, i) -> t*SVector(0.3, -0.05, 0.0)))

Fwall = viscous_force(:bottom, model, config_wall)
@test norm(Fwall) > 0
# direct sum of -ρ ν A (∇U_b + ∇U_bᵀ - ⅔ tr(∇U_b) I)⋅n over the patch faces, with ∇U_b the Gauss
# cell gradient whose normal part is the normal gradient of the wall condition, (U_w - U_c)/δ
vf_reference = function (patch, Ub, config)
    gradU = Grad{Gauss}(U)
    grad!(gradU, FaceVectorField(mesh_dev), U, config.boundaries.U, 0.0, config)
    IDs_range = mesh_dev.boundaries[boundary_index(mesh_dev.boundaries, patch)].IDs_range
    (; faces, boundary_cellsID) = mesh_dev
    sum(IDs_range) do fID
        cID, n = boundary_cellsID[fID], faces.normal[fID]
        Mc = gradU[cID]
        sngrad = Ub === nothing ? zero(U[cID]) : (Ub - U[cID])/faces.delta[fID] # zero gradient: 0
        Mb = Mc + (sngrad - Mc*n)*n'
        -model.fluid.rho[1]*1e-3*faces.area[fID]*((Mb + Mb' - (2/3)*tr(Mb)*I)*n)
    end
end
@test Fwall ≈ vf_reference(:bottom, SVector{3}(Uw), config_wall) rtol=1e-12
@test viscous_force(:bottom, model, config_dir) == Fwall
@test viscous_force(:bottom, model, config_fun) ≈ Fwall rtol=1e-12
# the user function is evaluated at the time given (default 0)
@test viscous_force(:bottom, model, config_funt; time=1.0) ≈ Fwall rtol=1e-12
@test viscous_force(:bottom, model, config_funt; time=2.0) ≈
    viscous_force(:bottom, model, config_wall2) rtol=1e-12
@test viscous_force(:bottom, model, config_funt) ≈
    viscous_force(:bottom, model, vf_config(Wall(:bottom, zero(Uw)))) rtol=1e-12

# RotatingWall against a DirichletFunction giving the same rotation velocity
centre, axis, rpm = SVector(0.0, -0.01, 0.0), SVector(0.0, 0.0, 1.0), 300.0
vf_rotation = (c, t, i) -> -(2π*rpm/60)*(c - centre) × axis
config_rot = vf_config(RotatingWall(:bottom, centre=collect(centre), axis=collect(axis), rpm=rpm))
config_rotf = vf_config(DirichletFunction(:bottom, vf_rotation))
Frot = viscous_force(:bottom, model, config_rot)
@test norm(Frot - Fwall) > 0
@test Frot ≈ viscous_force(:bottom, model, config_rotf) rtol=1e-12

# a zero-gradient patch has no normal gradient, but the transposed cell gradient acts on the face;
# a mirror (Slip, Symmetry) patch carries no shear
@test viscous_force(:top, model, config_wall) ≈ vf_reference(:top, nothing, config_wall) rtol=1e-12
config_slip = Configuration(
    solvers=(;), schemes=(;), runtime=Runtime(iterations=1, time_step=1, write_interval=-1),
    hardware=hardware, postprocess=nothing,
    boundaries=assign(region=mesh_dev, (
        U=[Zerogradient(:inlet), Zerogradient(:outlet), Wall(:bottom, Uw), Slip(:top)], p=pBCs)))
@test all(iszero, viscous_force(:top, model, config_slip))
@test all(iszero, wall_shear_stress(:top, model, config_slip)[1].x.values)

# exact linear fields with exact values on every patch, so the Gauss gradient is exact: pure
# shear U = (γy, 0) gives ρνγ per unit area; U = (0, βx) is only seen through the transposed term
vf_exact = function (Ufun)
    for i ∈ eachindex(U.x.values)
        u = Ufun(mesh_dev.cells.centre[i])
        U.x.values[i], U.y.values[i], U.z.values[i] = u
    end
    bc = name -> DirichletFunction(name, (c, t, i) -> Ufun(c))
    config = Configuration(
        solvers=(;), schemes=(;), runtime=Runtime(iterations=1, time_step=1, write_interval=-1),
        hardware=hardware, postprocess=nothing,
        boundaries=assign(region=mesh_dev, (U=[bc(:inlet), bc(:outlet), bc(:bottom), bc(:top)], p=pBCs)))
    IDs_range = mesh_dev.boundaries[boundary_index(mesh_dev.boundaries, :bottom)].IDs_range
    viscous_force(:bottom, model, config), sum(mesh_dev.faces.area[IDs_range])
end
Usaved = (copy(U.x.values), copy(U.y.values), copy(U.z.values))
γ, β = 150.0, 80.0
F, A = vf_exact(c -> SVector(γ*c[2], 0.0, 0.0))
@test F ≈ [model.fluid.rho[1]*1e-3*γ*A, 0.0, 0.0] rtol=1e-10 atol=1e-14
F, A = vf_exact(c -> SVector(0.0, β*c[1], 0.0))
@test F ≈ [model.fluid.rho[1]*1e-3*β*A, 0.0, 0.0] rtol=1e-10 atol=1e-14
U.x.values .= Usaved[1]; U.y.values .= Usaved[2]; U.z.values .= Usaved[3]

# wall_shear_stress uses the same face values; its integral is the viscous force
vf_shear_force = function (config; kw...)
    tauw, pos = wall_shear_stress(:bottom, model, config; kw...)
    IDs_range = mesh_dev.boundaries[boundary_index(mesh_dev.boundaries, :bottom)].IDs_range
    @test pos == [mesh_dev.faces.centre[fID] for fID ∈ IDs_range]
    area = mesh_dev.faces.area[IDs_range]
    model.fluid.rho[1] .* SVector(sum(tauw.x.values .* area), sum(tauw.y.values .* area),
        sum(tauw.z.values .* area))
end
@test vf_shear_force(config_wall) ≈ Fwall rtol=1e-12
@test vf_shear_force(config_fun) ≈ Fwall rtol=1e-12
@test vf_shear_force(config_funt; time=2.0) ≈ viscous_force(:bottom, model, config_wall2) rtol=1e-12
@test vf_shear_force(config_rot) ≈ Frot rtol=1e-12

# stress_tensor evaluates the velocity conditions at the time given (default 0)
nut0 = ScalarField(mesh_dev)
vf_R = (config; kw...) -> stress_tensor(U, 1e-3, nut0, config; kw...).xy.values
@test vf_R(config_funt; time=1.0) ≈ vf_R(config_wall) rtol=1e-12
@test vf_R(config_funt; time=2.0) ≈ vf_R(config_wall2) rtol=1e-12
@test !(vf_R(config_wall2) ≈ vf_R(config_wall))
