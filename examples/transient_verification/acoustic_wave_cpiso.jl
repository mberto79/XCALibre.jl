using XCALibre

# Acoustic standing wave in a periodic strip of air, p = 101325(1 + 0.01cos(2πx)), with cpiso!:
# compressible time terms at an acoustic Courant number of about 0.5. CrankNicolson takes several
# passes per step (outer_loops); error against CrankNicolson at Δt/32 (see README.md).

n = 128  # Mesh: 128 × 2 cells, periodic in x and y
points = [Point(0.0, 0.0, 0.0), Point(1.0, 0.0, 0.0), Point(0.0, 2/n, 0.0), Point(1.0, 2/n, 0.0)]
edges = [line!(points, 1, 2, n), line!(points, 3, 4, n),
    line!(points, 1, 3, 2), line!(points, 2, 4, 2)]
patches = [Patch(:left, [3]), Patch(:right, [4]), Patch(:bottom, [1]), Patch(:top, [2])]
blocks = [quad(edges, [1, 2, 3, 4])]
mesh = generate!(MeshBuilder2D(points, edges, patches, blocks))
mesh = XCALibre.UNV2.update_mesh_format(mesh, Int64, Float64)

backend = CPU(); workgroup = 1024
hardware = Hardware(backend=backend, workgroup=workgroup)

periodic_x = construct_periodic(mesh, backend, :left, :right)
periodic_y = construct_periodic(mesh, backend, :bottom, :top)

xc = [cell.centre[1] for cell ∈ mesh.cells]
p_inf = 101325.0
T_inf = 300.0
dt_coarse = 1.1e-5 # acoustic Courant number ≈ 0.5
tend = 64*dt_coarse

function run_case(timeScheme, dt; outer_loops)
    model = Physics(
        time = Transient(),
        fluid = Fluid{Compressible}(nu=1e-3, cp=1005.0, gamma=1.4, Pr=0.7),
        turbulence = RANS{Laminar}(),
        energy = Energy{SensibleEnthalpy}(Tref=0.0),
        domain = mesh
        )

    BCs = assign(
        region = mesh,
        (
            U = [periodic_x..., periodic_y...],
            p = [periodic_x..., periodic_y...],
            he = [periodic_x..., periodic_y...]
        )
    )

    schemes = (
        U = Schemes(time=timeScheme, divergence=Linear, gradient=Gauss),
        p = Schemes(time=timeScheme, gradient=Gauss),
        he = Schemes(time=timeScheme, divergence=Linear, gradient=Gauss)
    )

    # tight tolerances: the solver error must stay below the temporal error
    tight = SolverSetup(
        solver = Bicgstab(),
        preconditioner = Jacobi(),
        convergence = 1e-12,
        relax = 1.0,
        rtol = 1e-12,
        atol = 1e-15
    )
    solvers = (U = tight, p = tight, he = tight)

    runtime = Runtime(iterations=round(Int, tend/dt), write_interval=-1, time_step=dt)

    config = Configuration(
        solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

    initialise!(model.momentum.U, [0.0, 0.0, 0.0])
    model.momentum.p.values .= p_inf .* (1 .+ 0.01 .* cos.(2pi .* xc))
    initialise!(model.energy.T, T_inf)

    run!(model, config; progress=false, outer_loops=outer_loops)

    return copy(model.momentum.p.values)
end

# Scheme comparison: max error of p at t = tend against the reference, observed order
dts = [dt_coarse, dt_coarse/2, dt_coarse/4]
cn_outer_loops = 8
p_ref = run_case(CrankNicolson, dt_coarse/32; outer_loops=cn_outer_loops)
time_schemes = ["Euler" => (Euler, 1), "CrankNicolson" => (CrankNicolson, cn_outer_loops)]
errors = Dict{String,Vector{Float64}}()
orders = Dict{String,Vector{Float64}}()
for (name, (timeScheme, outer_loops)) ∈ time_schemes
    errors[name] = [maximum(abs.(run_case(timeScheme, dt; outer_loops) .- p_ref)) for dt ∈ dts]
    orders[name] = log2.(errors[name][1:end-1] ./ errors[name][2:end])
end

println("\nAcoustic wave (cpiso!), error of p [Pa] at t = $tend against a Δt/32 reference")
println("(Euler with outer_loops = 1, CrankNicolson with outer_loops = $cn_outer_loops)")
println(rpad("dt", 12), join((rpad(name, 30) for (name, _) ∈ time_schemes)))
for i ∈ eachindex(dts)
    row = map(time_schemes) do (name, _)
        o = i > 1 ? "  (order $(round(orders[name][i-1], digits=2)))" : ""
        rpad("$(round(errors[name][i], sigdigits=3))$o", 30)
    end
    println(rpad(round(dts[i], sigdigits=3), 12), join(row))
end

results = (; dts, errors, orders)
