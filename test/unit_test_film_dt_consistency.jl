# A film run to a steady state must not depend on Δt: inclined plate, prescribed h and U at
# the inlet, σ = 0. Main fef95b73 gave max|Δh|/h_N = 1.06e-3 and max|ΔUx|/U_N = 5.5e-4
# between Δt = 5e-5 and 6e-4; this fix gives 5.8e-5 and 5.7e-7.
function film_dt_steady(dt; tend=1.0, σ=0.0)
    nx, ny = 50, 4
    L, W = 0.05, 0.004
    P = XCALibre.Point
    pts = [P(0.0, 0.0, 0.0), P(L, 0.0, 0.0), P(0.0, W, 0.0), P(L, W, 0.0)]
    edges = [XCALibre.line!(pts, 1, 2, nx), XCALibre.line!(pts, 3, 4, nx),
        XCALibre.line!(pts, 1, 3, ny), XCALibre.line!(pts, 2, 4, ny)]
    patches = [XCALibre.Patch(:inlet, [3]), XCALibre.Patch(:outlet, [4]),
        XCALibre.Patch(:side_1, [1]), XCALibre.Patch(:side_2, [2])]
    builder = XCALibre.MeshBuilder2D(pts, edges, patches, [XCALibre.quad(edges, [1, 2, 3, 4])])
    mesh = XCALibre.UNV2.update_mesh_format(XCALibre.generate!(builder), Int64, Float64)

    nu, rho_l, g, ϕ = 0.001003/998.2, 998.2, 9.81, 30
    q = 25.53e-5/0.51
    hN = (3*nu*q/(g*sind(ϕ)))^(1/3)
    UN = q/hN

    model = Physics(
        momentum = Momentum{EFM}(; σ=σ, β=1, θm=75, gravity=(g*sind(ϕ), 0.0, -g*cosd(ϕ))),
        time = Transient(),
        fluid = Fluid{Incompressible}(; nu=nu, rho=rho_l),
        turbulence = RANS{Laminar}(),
        energy = Energy{Isothermal}(),
        domain = mesh
        )
    BCs = assign(region=mesh, (
        U = [Dirichlet(:inlet, [UN/1.5, 0.0, 0.0]), Extrapolated(:outlet),
            Extrapolated(:side_1), Extrapolated(:side_2)],
        h = [Dirichlet(:inlet, 1.5*hN), Extrapolated(:outlet),
            Extrapolated(:side_1), Extrapolated(:side_2)]
        ))
    solver() = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-11,
        relax=1.0, rtol=0.0, atol=1e-12)
    config = Configuration(
        solvers = (U=solver(), h=solver()),
        schemes = (U=Schemes(time=Euler, divergence=Upwind),
            h=Schemes(time=Euler, divergence=LUST)),
        runtime = Runtime(iterations=round(Int, tend/dt), time_step=dt, write_interval=-1),
        hardware = Hardware(backend=CPU(), workgroup=1024),
        boundaries = BCs)

    initialise!(model.momentum.U, [UN, 0.0, 0.0])
    initialise!(model.momentum.h, hN)
    run!(model, config, inner_loops=2, progress=false)
    return copy(model.momentum.h.values)./hN, copy(model.momentum.U.x.values)./UN
end

@testset "Film steady state independent of Δt" begin
    h1, u1 = film_dt_steady(5e-5)
    h2, u2 = film_dt_steady(6e-4)
    @test all(isfinite, h1) && all(isfinite, h2)
    @test maximum(abs.(h1 .- h2)) < 2e-4
    @test maximum(abs.(u1 .- u2)) < 1e-5
end
