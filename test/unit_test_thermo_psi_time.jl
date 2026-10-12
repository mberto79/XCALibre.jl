using XCALibre
using Test

# The face compressibility must evaluate time-dependent energy boundary values at the solver time
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "laplace_unit_3by3.unv"))
backend = CPU()
mesh_dev = adapt(backend, mesh)
hardware = Hardware(backend=backend, workgroup=1024)

cp = 1005.0; gamma = 1.4; R = cp*(1 - 1/gamma)
Tref = 0.0
T_wall(t) = 300.0 + 10.0*t
he_wall(coords, time, index) = cp*(T_wall(time) - Tref)

for energy in (Energy{SensibleEnthalpy}(Tref=Tref), Energy{InternalEnergy}(Tref=Tref))
    model = Physics(
        time = Steady(),
        fluid = Fluid{Compressible}(nu=1e-5, cp=cp, gamma=gamma, Pr=0.7),
        turbulence = RANS{Laminar}(),
        energy = energy,
        domain = mesh_dev)

    BCs = assign(region=mesh_dev, (
        U = [Wall(:left_wall, [0.0, 0.0, 0.0]), Wall(:right_wall, [0.0, 0.0, 0.0]),
             Wall(:bottom_wall, [0.0, 0.0, 0.0]), Wall(:upper_wall, [0.0, 0.0, 0.0])],
        p = [Zerogradient(:left_wall), Zerogradient(:right_wall),
             Zerogradient(:bottom_wall), Zerogradient(:upper_wall)],
        he = [DirichletFunction(:left_wall, he_wall), Zerogradient(:right_wall),
              Zerogradient(:bottom_wall), Zerogradient(:upper_wall)]))

    config = Configuration(
        solvers=(;), schemes=(;), boundaries=BCs, hardware=hardware,
        runtime=Runtime(iterations=1, write_interval=-1, time_step=1))

    initialise!(model.energy.T, 300.0)
    XCALibre.ModelPhysics.temperature_to_energy!(model, model.energy.T, model.energy.he)
    Psif = FaceScalarField(mesh_dev)
    time = 5.0
    XCALibre.ModelPhysics.thermo_Psi!(model, Psif, time, config)

    wall = mesh.boundaries[findfirst(b -> b.name == :left_wall, mesh.boundaries)].IDs_range
    # Sensible enthalpy: he = cp*T; internal energy: the same functor value read as cv*T
    T_expected = energy isa Energy{SensibleEnthalpy} ? T_wall(time) : gamma*T_wall(time)
    @test all(isapprox.(Psif.values[wall], 1/(R*T_expected); rtol=1e-12))
end
