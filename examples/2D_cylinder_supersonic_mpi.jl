# Distributed (MPI) steady Mach 1.5 laminar cylinder with the explicit density-based solver and
# local time stepping; prints the density residual and the drag history. Needs XCALibre, PETSc, MPI.
# Run: mpiexecjl -n 4 julia --project=<env> examples/2D_cylinder_supersonic_mpi.jl
using XCALibre, PETSc, MPI

# every rank makes this identical call: rank 0 reads and partitions, the others receive
mesh_dist = distribute() do
    UNV2D_mesh(joinpath(pkgdir(XCALibre, "examples/0_GRIDS"), "cylinder_d10mm_5mm.unv"), scale=0.001)
end

hardware = Hardware(backend=CPU(), workgroup=64)

gamma = 1.4; cp = 1005.0; Pr = 0.7; nu = 1e-5
T_inf = 300.0; p_inf = 101325.0
U_inf = 1.5 * sqrt(gamma * cp * (1.0 - 1.0/gamma) * T_inf)
velocity = [U_inf, 0.0, 0.0]

model = Physics(
    time = Transient(),
    fluid = Fluid{SupersonicFlow}(nu=nu, cp=cp, gamma=gamma, Pr=Pr),
    turbulence = RANS{Laminar}(),
    energy = Energy{SensibleEnthalpy}(Tref=0.0),
    domain = mesh_dist)

BCs = assign(region=mesh_dist, (
    U = [Dirichlet(:inlet, velocity), Zerogradient(:outlet),
         Wall(:cylinder, [0.0, 0.0, 0.0]), Slip(:top), Slip(:bottom)],
    p = [Dirichlet(:inlet, p_inf), Zerogradient(:outlet), Zerogradient(:cylinder),
         Zerogradient(:top), Zerogradient(:bottom)],
    T = [Dirichlet(:inlet, T_inf), Zerogradient(:outlet), Zerogradient(:cylinder),
         Zerogradient(:top), Zerogradient(:bottom)]))

schemes = (
    U = Schemes(gradient=Gauss), p = Schemes(gradient=Gauss), T = Schemes(gradient=Gauss),
    flux = HLLC(), time_stepping = RK2(), reconstruction = MUSCL{VanLeer}())
runtime = Runtime(iterations=2000, write_interval=-1, time_step=1e-7,
    adaptive=LocalTimeStepping(maxCo=0.5, startCo=0.1, ramp=200))
config = Configuration(solvers=(rho=(convergence=1e-15,),), schemes=schemes,
    runtime=runtime, hardware=hardware, boundaries=BCs)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, p_inf)
initialise!(model.energy.T, T_inf)

# residuals and forces are reduced over the ranks, so every rank holds the same values
res = run!(model, config; progress=false, forces=(patches=[:cylinder], interval=500))

if is_root()
    println("density residual at iterations 500/1000/2000: ", round.(res.rho[[500, 1000, 2000]], sigdigits=3))
    for (it, F) ∈ zip(res.forces.cylinder.iteration, res.forces.cylinder.pressure)
        println("iteration $it: x pressure force on the cylinder ", round(F[1], sigdigits=6))
    end
end
