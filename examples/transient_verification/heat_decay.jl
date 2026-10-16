using XCALibre

# Transient conduction (laplace!): decay of T = 1 + cos(πx) on a 1-D strip with zero-gradient ends.
# cos(πxᵢ) is an eigenvector of the discrete Laplacian (eigenvalue λₕ), so the error against
# 1 + cos(πx)exp(λₕt) is purely temporal; also runs the off-centred CrankNicolson(0.9).

nx = 20  # Mesh: 20 × 1 cells
L = 1.0
points = [Point(0.0, 0.0, 0.0), Point(L, 0.0, 0.0), Point(0.0, 0.1, 0.0), Point(L, 0.1, 0.0)]
edges = [line!(points, 1, 2, nx), line!(points, 3, 4, nx),
    line!(points, 1, 3, 1), line!(points, 2, 4, 1)]
patches = [Patch(:left, [3]), Patch(:right, [4]), Patch(:bottom, [1]), Patch(:top, [2])]
blocks = [quad(edges, [1, 2, 3, 4])]
mesh = generate!(MeshBuilder2D(points, edges, patches, blocks))
mesh = XCALibre.UNV2.update_mesh_format(mesh, Int64, Float64)

backend = CPU(); workgroup = 1024
hardware = Hardware(backend=backend, workgroup=workgroup)

# Exact semi-discrete solution
xc = [cell.centre[1] for cell ∈ mesh.cells]
dx = L/nx
λh = -(2 - 2cos(pi*dx))/dx^2
tend = 0.05
T_exact = 1 .+ cos.(pi .* xc) .* exp(λh*tend)

function run_case(timeScheme, dt)
    model = Physics(
        time = Transient(),
        solid = Solid{Uniform}(k=1.0, rho=1.0, cp=1.0),
        energy = Energy{Conduction}(),
        domain = mesh
        )

    BCs = assign(
        region = mesh,
        (
            T = [
                Zerogradient(:left),
                Zerogradient(:right),
                Zerogradient(:bottom),
                Zerogradient(:top)
            ],
        )
    )

    schemes = Schemes(time=timeScheme, laplacian=Linear)

    # tight tolerances: the solver error must stay below the temporal error
    solvers = SolverSetup(
        solver = Cg(),
        preconditioner = Jacobi(),
        convergence = 1e-12,
        relax = 1.0,
        rtol = 1e-12,
        atol = 1e-15
    )

    runtime = Runtime(iterations=round(Int, tend/dt), write_interval=-1, time_step=dt)

    config = Configuration(
        solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

    model.energy.T.values .= 1 .+ cos.(pi .* xc) # initialise! takes a uniform value only

    run!(model, config; progress=false)

    return copy(model.energy.T.values)
end

# Scheme comparison: max error at t = tend against the exact solution, observed order
dts = [0.005, 0.0025, 0.00125, 0.000625]
time_schemes = ["Euler" => Euler, "CrankNicolson(0.9)" => CrankNicolson(0.9),
    "CrankNicolson" => CrankNicolson]
errors = Dict{String,Vector{Float64}}()
orders = Dict{String,Vector{Float64}}()
for (name, timeScheme) ∈ time_schemes
    errors[name] = [maximum(abs.(run_case(timeScheme, dt) .- T_exact)) for dt ∈ dts]
    orders[name] = log2.(errors[name][1:end-1] ./ errors[name][2:end])
end

println("\nTransient conduction, error at t = $tend against the exact solution")
println(rpad("dt", 12), join((rpad(name, 30) for (name, _) ∈ time_schemes)))
for i ∈ eachindex(dts)
    row = map(time_schemes) do (name, _)
        o = i > 1 ? "  (order $(round(orders[name][i-1], digits=2)))" : ""
        rpad("$(round(errors[name][i], sigdigits=3))$o", 30)
    end
    println(rpad(dts[i], 12), join(row))
end

results = (; dts, errors, orders)
