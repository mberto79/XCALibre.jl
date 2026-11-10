using XCALibre

# Decaying shear wave u = sin(2πy) with LES{Smagorinsky} (piso!): ν_t varies in space and time,
# so CrankNicolson must extrapolate it to tⁿ⁺¹. Error against CrankNicolson at Δt/16.
# (WALE gives ν_t = 0 in pure shear.)

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

yc = [cell.centre[2] for cell ∈ mesh.cells]
tend = 1.0

function run_case(timeScheme, dt)
    model = Physics(
        time = Transient(),
        fluid = Fluid{Incompressible}(nu = 1e-3),
        turbulence = LES{Smagorinsky}(C=1.0),
        energy = Energy{Isothermal}(),
        domain = mesh
        )

    BCs = assign(
        region = mesh,
        (
            U = [periodic_x..., periodic_y...],
            p = [periodic_x..., periodic_y...],
            nut = [periodic_x..., periodic_y...]
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
    initialise!(model.turbulence.nut, 0.0) # ν_t of the initial field is computed on step 1

    run!(model, config; progress=false, pref=0.0)

    return copy(U.x.values)
end

# Scheme comparison: max error at t = tend against the reference, observed order
dts = [0.05, 0.025, 0.0125]
Ux_ref = run_case(CrankNicolson, dts[end]/16)
time_schemes = ["Euler" => Euler, "CrankNicolson" => CrankNicolson]
errors = Dict{String,Vector{Float64}}()
orders = Dict{String,Vector{Float64}}()
for (name, timeScheme) ∈ time_schemes
    errors[name] = [maximum(abs.(run_case(timeScheme, dt) .- Ux_ref)) for dt ∈ dts]
    orders[name] = log2.(errors[name][1:end-1] ./ errors[name][2:end])
end

println("\nShear wave with Smagorinsky (piso!), error of Ux at t = $tend against a Δt/16 reference")
println(rpad("dt", 12), join((rpad(name, 30) for (name, _) ∈ time_schemes)))
for i ∈ eachindex(dts)
    row = map(time_schemes) do (name, _)
        o = i > 1 ? "  (order $(round(orders[name][i-1], digits=2)))" : ""
        rpad("$(round(errors[name][i], sigdigits=3))$o", 30)
    end
    println(rpad(dts[i], 12), join(row))
end

results = (; dts, errors, orders)
