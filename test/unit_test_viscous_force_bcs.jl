using XCALibre
using StaticArrays
using LinearAlgebra
# viscous_force and wall_shear_stress take the wall velocity of each face from the velocity
# condition of the patch: a fixed vector (Wall, Dirichlet), a user function evaluated at the
# face centre and `time` (DirichletFunction) or the rotation velocity (RotatingWall).

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
config_for(bottom) = Configuration(
    solvers=(;), schemes=(;), runtime=Runtime(iterations=1, time_step=1, write_interval=-1),
    hardware=hardware, postprocess=nothing,
    boundaries=assign(region=mesh_dev, (
        U=[Zerogradient(:inlet), Zerogradient(:outlet), bottom, Zerogradient(:top)], p=pBCs)))

Uw = [0.3, -0.05, 0.0]
config_wall = config_for(Wall(:bottom, Uw))
config_wall2 = config_for(Wall(:bottom, 2Uw))
config_dir = config_for(Dirichlet(:bottom, Uw))
config_fun = config_for(DirichletFunction(:bottom, (c, t, i) -> SVector(0.3, -0.05, 0.0)))
config_funt = config_for(DirichletFunction(:bottom, (c, t, i) -> t*SVector(0.3, -0.05, 0.0)))

Fwall = viscous_force(:bottom, model, config_wall)
@test norm(Fwall) > 0
# direct sum of ρ ν A (U_c - U_w)_t/δ over the patch faces
let IDs_range = mesh_dev.boundaries[boundary_index(mesh_dev.boundaries, :bottom)].IDs_range
    (; faces, boundary_cellsID) = mesh_dev
    F = sum(IDs_range) do fID
        n, d = faces.normal[fID], U[boundary_cellsID[fID]] - Uw
        model.fluid.rho[1]*1e-3*faces.area[fID]*(d - (d⋅n)*n)/faces.delta[fID]
    end
    @test Fwall ≈ F rtol=1e-12
end
@test viscous_force(:bottom, model, config_dir) == Fwall
@test viscous_force(:bottom, model, config_fun) ≈ Fwall rtol=1e-12
# the user function is evaluated at the time given (default 0)
@test viscous_force(:bottom, model, config_funt; time=1.0) ≈ Fwall rtol=1e-12
@test viscous_force(:bottom, model, config_funt; time=2.0) ≈
    viscous_force(:bottom, model, config_wall2) rtol=1e-12
@test viscous_force(:bottom, model, config_funt) ≈
    viscous_force(:bottom, model, config_for(Wall(:bottom, zero(Uw)))) rtol=1e-12

# RotatingWall against a DirichletFunction giving the same rotation velocity
centre, axis, rpm = SVector(0.0, -0.01, 0.0), SVector(0.0, 0.0, 1.0), 300.0
rotation(c, t, i) = -(2π*rpm/60)*(c - centre) × axis
config_rot = config_for(RotatingWall(:bottom, centre=collect(centre), axis=collect(axis), rpm=rpm))
config_rotf = config_for(DirichletFunction(:bottom, rotation))
Frot = viscous_force(:bottom, model, config_rot)
@test norm(Frot - Fwall) > 0
@test Frot ≈ viscous_force(:bottom, model, config_rotf) rtol=1e-12

# a zero-gradient patch carries no shear
@test all(iszero, viscous_force(:top, model, config_wall))

# wall_shear_stress uses the same face values; its integral is the viscous force
function shear_force(config; kw...)
    tauw, pos = wall_shear_stress(:bottom, model, config; kw...)
    IDs_range = mesh_dev.boundaries[boundary_index(mesh_dev.boundaries, :bottom)].IDs_range
    @test pos == [mesh_dev.faces.centre[fID] for fID ∈ IDs_range]
    area = mesh_dev.faces.area[IDs_range]
    model.fluid.rho[1] .* SVector(sum(tauw.x.values .* area), sum(tauw.y.values .* area),
        sum(tauw.z.values .* area))
end
@test shear_force(config_wall) ≈ Fwall rtol=1e-12
@test shear_force(config_fun) ≈ Fwall rtol=1e-12
@test shear_force(config_funt; time=2.0) ≈ viscous_force(:bottom, model, config_wall2) rtol=1e-12
@test shear_force(config_rot) ≈ Frot rtol=1e-12

# stress_tensor evaluates the velocity conditions at the time given (default 0)
nut0 = ScalarField(mesh_dev)
R(config; kw...) = stress_tensor(U, 1e-3, nut0, config; kw...).xy.values
@test R(config_funt; time=1.0) ≈ R(config_wall) rtol=1e-12
@test R(config_funt; time=2.0) ≈ R(config_wall2) rtol=1e-12
@test !(R(config_wall2) ≈ R(config_wall))
