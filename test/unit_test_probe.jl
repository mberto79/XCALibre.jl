using XCALibre
using Test

# A Probe samples the cell nearest to its location every step and writes one row per step.

@testset "Probe" begin
    mesh = UNV2D_mesh(joinpath(pkgdir(XCALibre, "examples/0_GRIDS"), "quad40.unv"), scale=0.001)
    model = Physics(time=Transient(), fluid=Fluid{Incompressible}(nu=1e-3),
        turbulence=RANS{Laminar}(), energy=Energy{Isothermal}(), domain=mesh)
    BCs = assign(region=mesh, (
        U = [Dirichlet(:inlet, [0.3, 0.0, 0.0]), Extrapolated(:outlet),
             Symmetry(:bottom, [0.0, 0.0, 0.0]), Symmetry(:top, [0.0, 0.0, 0.0])],
        p = [Extrapolated(:inlet), Dirichlet(:outlet, 0.0), Symmetry(:bottom), Symmetry(:top)]))
    setup(solver) = SolverSetup(solver=solver, preconditioner=Jacobi(), convergence=1e-7,
        relax=1.0, rtol=1e-6)
    solvers = (U=setup(Bicgstab()), p=setup(Cg()))
    schemes = (U=Schemes(time=Euler), p=Schemes())
    location = [0.0123, 0.0456, 0.0]
    nsteps = 5
    dir = mktempdir()
    probes = cd(dir) do
        pU = Probe(model.momentum.U, mesh; location=location, name="probe_U")
        pp = Probe(model.momentum.p, mesh; location=location, name="probe_p", output=TXT())
        config = Configuration(solvers=solvers, schemes=schemes,
            runtime=Runtime(iterations=nsteps, time_step=0.01, write_interval=-1),
            hardware=Hardware(backend=CPU(), workgroup=64), boundaries=BCs, postprocess=[pU, pp])
        initialise!(model.momentum.U, [0.3, 0.0, 0.0])
        initialise!(model.momentum.p, 0.0)
        run!(model, config; progress=false)
        pU, pp
    end

    d2(c) = sum(abs2, c .- location)
    nearest = argmin(i -> d2(mesh.cells[i].centre), eachindex(mesh.cells))
    @test all(p.index == nearest for p ∈ probes)

    rowsU = split.(readlines(joinpath(dir, "probe_U.csv"))[2:end], ',')
    rowsp = split.(readlines(joinpath(dir, "probe_p.txt")), ' ')
    @test length(rowsU) == length(rowsp) == nsteps
    (; U, p) = model.momentum
    @test parse.(Float64, rowsU[end][2:4]) == [U.x.values[nearest], U.y.values[nearest], U.z.values[nearest]]
    @test parse(Float64, rowsp[end][2]) == p.values[nearest]
end
