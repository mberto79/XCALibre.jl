# n=1 smoke: BFS laminar psimple! runs + tracks serial (converged compare in gate file)
using XCALibre, PETSc, MPI

MPI.Init()

iters = parse(Int, get(ENV, "XCAL_SMOKE_ITERS", "5"))
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "backwardFacingStep_10mm.unv"), scale=0.001)
velocity = [0.5, 0.0, 0.0]

function bfs_case(mesh, iters)
    model = Physics(
        time = Steady(),
        fluid = Fluid{Incompressible}(nu=1e-3),
        turbulence = RANS{Laminar}(),
        energy = Energy{Isothermal}(),
        domain = mesh)
    BCs = assign(region=mesh, (
        U = [Dirichlet(:inlet, [0.5, 0.0, 0.0]), Extrapolated(:outlet),
             Wall(:wall, [0.0, 0.0, 0.0]), Symmetry(:top)],
        p = [Extrapolated(:inlet), Dirichlet(:outlet, 0.0),
             Extrapolated(:wall), Symmetry(:top)]))
    conv = parse(Float64, get(ENV, "XCAL_CONV", "1e-7"))
    rtolU = parse(Float64, get(ENV, "XCAL_RTOL_U", "1e-1"))
    rtolp = parse(Float64, get(ENV, "XCAL_RTOL_P", "1e-2"))
    solvers = (
        U = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(),
            convergence=conv, relax=0.8, rtol=rtolU, atol=1e-12),
        p = SolverSetup(solver=Cg(), preconditioner=Jacobi(),
            convergence=conv, relax=0.2, rtol=rtolp, atol=1e-12))
    schemes = (U=Schemes(divergence=Linear), p=Schemes())
    runtime = Runtime(iterations=iters, time_step=1, write_interval=-1)
    config = Configuration(solvers=solvers, schemes=schemes, runtime=runtime,
        hardware=Hardware(backend=CPU(), workgroup=64), boundaries=BCs)
    initialise!(model.momentum.U, velocity)
    initialise!(model.momentum.p, 0.0)
    model, config
end

model_s, config_s = bfs_case(mesh, iters)
Rs = simple!(model_s, config_s)

dm = distribute(mesh)
model_d, config_d = bfs_case(dm, iters)
Rd = prun!(model_d, config_d)

n = dm.partition.n_owned
orig = dm.orig_cells
dp = maximum(abs.(model_d.momentum.p.values[1:n] .- model_s.momentum.p.values[orig[1:n]]))
du = maximum(abs.(model_d.momentum.U.x.values[1:n] .- model_s.momentum.U.x.values[orig[1:n]]))
println("SMOKE iters=$iters  Rp_serial=$(Rs.p[iters])  Rp_dist=$(Rd.p[iters])  dp=$dp  du=$du")
for k ∈ (1, 2, 5, 10, 20, 50, 100, iters)
    k <= iters && println("  it=$k  Rux s/d = $(Rs.Ux[k]) / $(Rd.Ux[k])   Ruy s/d = $(Rs.Uy[k]) / $(Rd.Uy[k])   Rp s/d = $(Rs.p[k]) / $(Rd.p[k])")
end
println(all(isfinite, model_d.momentum.p.values) && all(isfinite, model_d.momentum.U.x.values) ?
    "SMOKE PASS" : "SMOKE FAIL nonfinite")
