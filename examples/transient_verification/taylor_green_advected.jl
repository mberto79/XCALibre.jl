using XCALibre

# Taylor-Green vortex carried by a uniform stream through a doubly periodic box, ν = 0.01 (piso!):
# nonlinear convection and pressure coupling, so the convecting face flux must be at the right
# time level. On 24² cells space errors dominate: error against CrankNicolson at Δt/16.

n = 24  # Mesh: 24 × 24 cells, periodic in x and y
points = [Point(0.0, 0.0, 0.0), Point(1.0, 0.0, 0.0), Point(0.0, 1.0, 0.0), Point(1.0, 1.0, 0.0)]
edges = [line!(points, 1, 2, n), line!(points, 3, 4, n),
    line!(points, 1, 3, n), line!(points, 2, 4, n)]
patches = [Patch(:left, [3]), Patch(:right, [4]), Patch(:bottom, [1]), Patch(:top, [2])]
blocks = [quad(edges, [1, 2, 3, 4])]
mesh = generate!(MeshBuilder2D(points, edges, patches, blocks))
mesh = XCALibre.UNV2.update_mesh_format(mesh, Int64, Float64)

backend = CPU(); workgroup = 1024
hardware = Hardware(backend=backend, workgroup=workgroup)

periodic_x = construct_periodic(mesh, backend, :left, :right)
periodic_y = construct_periodic(mesh, backend, :bottom, :top)

xc = [cell.centre[1] for cell ∈ mesh.cells]
yc = [cell.centre[2] for cell ∈ mesh.cells]
tend = 0.2

function run_case(timeScheme, dt)
    model = Physics(
        time = Transient(),
        fluid = Fluid{Incompressible}(nu = 0.01),
        turbulence = RANS{Laminar}(),
        energy = Energy{Isothermal}(),
        domain = mesh
        )

    BCs = assign(
        region = mesh,
        (
            U = [periodic_x..., periodic_y...],
            p = [periodic_x..., periodic_y...]
        )
    )

    schemes = (
        U = Schemes(time=timeScheme, divergence=Linear, gradient=Gauss),
        p = Schemes(time=timeScheme, gradient=Gauss)
    )

    # tight tolerances: the solver error must stay below the temporal error
    solvers = (
        U = SolverSetup(
            solver = Bicgstab(),
            preconditioner = Jacobi(),
            convergence = 1e-12,
            relax = 1.0,
            rtol = 1e-12,
            atol = 1e-15
        ),
        p = SolverSetup(
            solver = Cg(),
            preconditioner = Jacobi(),
            convergence = 1e-12,
            relax = 1.0,
            rtol = 1e-12,
            atol = 1e-15
        )
    )

    runtime = Runtime(iterations=round(Int, tend/dt), write_interval=-1, time_step=dt)

    config = Configuration(
        solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

    U = model.momentum.U
    U.x.values .= 1.0 .- cos.(2pi .* xc) .* sin.(2pi .* yc)
    U.y.values .= 0.5 .+ sin.(2pi .* xc) .* cos.(2pi .* yc)
    U.z.values .= 0.0
    initialise!(model.momentum.p, 0.0)

    run!(model, config; progress=false, pref=0.0)

    return [copy(U.x.values); copy(U.y.values)]
end

# Scheme comparison: max error of (Ux, Uy) at t = tend against the reference, observed order
dts = [0.02, 0.01, 0.005]
U_ref = run_case(CrankNicolson, dts[end]/16)
time_schemes = ["Euler" => Euler, "CrankNicolson" => CrankNicolson]
errors = Dict{String,Vector{Float64}}()
orders = Dict{String,Vector{Float64}}()
for (name, timeScheme) ∈ time_schemes
    errors[name] = [maximum(abs.(run_case(timeScheme, dt) .- U_ref)) for dt ∈ dts]
    orders[name] = log2.(errors[name][1:end-1] ./ errors[name][2:end])
end

println("\nAdvected Taylor-Green vortex (piso!), error of U at t = $tend against a Δt/16 reference")
println(rpad("dt", 12), join((rpad(name, 30) for (name, _) ∈ time_schemes)))
for i ∈ eachindex(dts)
    row = map(time_schemes) do (name, _)
        o = i > 1 ? "  (order $(round(orders[name][i-1], digits=2)))" : ""
        rpad("$(round(errors[name][i], sigdigits=3))$o", 30)
    end
    println(rpad(dts[i], 12), join(row))
end

results = (; dts, errors, orders)
