# Compile cost of a second mesh size in one session: laminar SIMPLE on two box meshes of different size, then the first again.
using XCALibre
grids = pkgdir(XCALibre, "examples/0_GRIDS")

function box_run(file)
    mesh = UNV3D_mesh(joinpath(grids, file), scale=0.001)
    model = Physics(time=Steady(), fluid=Fluid{Incompressible}(nu=1e-3), turbulence=RANS{Laminar}(),
        energy=Energy{Isothermal}(), domain=mesh)
    walls = (:y_min, :y_max, :z_min, :z_max)
    bcs = assign(region=mesh, (
        U = [Dirichlet(:x_min, [0.5, 0.0, 0.0]), Extrapolated(:x_max), [Wall(w, [0.0, 0.0, 0.0]) for w ∈ walls]...],
        p = [Extrapolated(:x_min), Dirichlet(:x_max, 0.0), [Extrapolated(w) for w ∈ walls]...]))
    solvers = (U=SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-15, relax=0.7, rtol=1e-8),
        p=SolverSetup(solver=Cg(), preconditioner=Jacobi(), convergence=1e-15, relax=0.3, rtol=1e-8))
    config = Configuration(solvers=solvers, schemes=(U=Schemes(divergence=Upwind), p=Schemes()),
        runtime=Runtime(iterations=3, write_interval=-1, time_step=1),
        hardware=Hardware(backend=CPU(), workgroup=AutoTune()), boundaries=bcs)
    initialise!(model.momentum.U, [0.0, 0.0, 0.0]); initialise!(model.momentum.p, 0.0)
    @elapsed run!(model, config)
end

a = box_run("3d_box_1000x1000x1000mm_5.unv")
b = box_run("3d_box_1000x1000x1000mm_10.unv")
c = box_run("3d_box_1000x1000x1000mm_5.unv")
println("COMPILE_SIZES first_s=$(round(a; digits=2)) second_size_s=$(round(b; digits=2)) warm_s=$(round(c; digits=3))")
