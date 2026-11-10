using XCALibre

# Decaying shear wave u = sin(2πy) in a doubly periodic box (piso!): convection and pressure vanish
# and sin(2πyⱼ) is an eigenvector of the discrete Laplacian (eigenvalue λₕ), so the error against
# sin(2πy)exp(λₕνt) is purely temporal; exercises the PISO predictor and correctors.

ny = 20  # Mesh: 2 × 20 cells, periodic in x and y
points = [Point(0.0, 0.0, 0.0), Point(0.1, 0.0, 0.0), Point(0.0, 1.0, 0.0), Point(0.1, 1.0, 0.0)]
edges = [line!(points, 1, 2, 2), line!(points, 3, 4, 2),
    line!(points, 1, 3, ny), line!(points, 2, 4, ny)]
patches = [Patch(:left, [3]), Patch(:right, [4]), Patch(:bottom, [1]), Patch(:top, [2])]
blocks = [quad(edges, [1, 2, 3, 4])]
mesh = generate!(MeshBuilder2D(points, edges, patches, blocks))
mesh = XCALibre.UNV2.update_mesh_format(mesh, Int64, Float64)

backend = CPU(); workgroup = 1024
hardware = Hardware(backend=backend, workgroup=workgroup)

periodic_x = construct_periodic(mesh, backend, :left, :right)
periodic_y = construct_periodic(mesh, backend, :bottom, :top)

# Exact semi-discrete solution
nu = 1.0
yc = [cell.centre[2] for cell ∈ mesh.cells]
dy = 1/ny
λh = -(2 - 2cos(2pi*dy))/dy^2
tend = 0.05
Ux_exact = sin.(2pi .* yc) .* exp(λh*nu*tend)

function run_case(timeScheme, dt)
    model = Physics(
        time = Transient(),
        fluid = Fluid{Incompressible}(nu = nu),
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
    U.x.values .= sin.(2pi .* yc); U.y.values .= 0.0; U.z.values .= 0.0
    initialise!(model.momentum.p, 0.0)

    # pref: the pressure level of a fully periodic domain is free
    run!(model, config; progress=false, pref=0.0, inner_loops=2)

    return copy(U.x.values)
end

# Scheme comparison: max error at t = tend against the exact solution, observed order
dts = [0.005, 0.0025, 0.00125, 0.000625]
time_schemes = ["Euler" => Euler, "CrankNicolson" => CrankNicolson]
errors = Dict{String,Vector{Float64}}()
orders = Dict{String,Vector{Float64}}()
for (name, timeScheme) ∈ time_schemes
    errors[name] = [maximum(abs.(run_case(timeScheme, dt) .- Ux_exact)) for dt ∈ dts]
    orders[name] = log2.(errors[name][1:end-1] ./ errors[name][2:end])
end

println("\nDecaying shear wave (piso!), error of Ux at t = $tend against the exact solution")
println(rpad("dt", 12), join((rpad(name, 30) for (name, _) ∈ time_schemes)))
for i ∈ eachindex(dts)
    row = map(time_schemes) do (name, _)
        o = i > 1 ? "  (order $(round(orders[name][i-1], digits=2)))" : ""
        rpad("$(round(errors[name][i], sigdigits=3))$o", 30)
    end
    println(rpad(dts[i], 12), join(row))
end

results = (; dts, errors, orders)
