# Assembly benchmark of the convection term: time, allocations and compile time per divergence scheme.
#
#   julia --project=<tree> [-t N] benchmark/divergence_assembly.jl [mesh.unv] [reps]
#
# Run it on two checkouts (baseline and branch) and compare the tables. Schemes that the checkout
# does not define are skipped. Timings are the minimum of `reps` warmed calls of `discretise!`.
using XCALibre, Statistics, Printf

mesh_file = length(ARGS) >= 1 && !isempty(ARGS[1]) ? ARGS[1] :
    pkgdir(XCALibre, "examples/0_GRIDS/backwardFacingStep_2mm.unv")
reps = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 50

mesh = UNV2D_mesh(mesh_file, scale=0.001)
backend = CPU()
hardware = Hardware(backend=backend, workgroup=1024)
mesh = adapt(backend, mesh)
ncells = length(mesh.cells)
config = Configuration(
    solvers=nothing, schemes=nothing, boundaries=nothing, hardware=hardware,
    runtime=Runtime(iterations=1, write_interval=-1, time_step=1))

mdotf = FaceScalarField(mesh)
mdotf.values .= 1e-3 .* (1 .+ 0.1 .* rand(length(mdotf.values)))
U = VectorField(mesh); U.x.values .= rand(ncells)
k = ScalarField(mesh); k.values .= rand(ncells)
gradU = Grad{Gauss}(U)
gradk = Grad{Gauss}(k)
nueff = FaceScalarField(mesh); nueff.values .= 1e-3

# the gradient argument exists only on branches with second-order schemes
div_term(S, flux, phi, grad) = hasmethod(Divergence{S}, Tuple{typeof(flux),typeof(phi),typeof(grad.result)}) ?
    Divergence{S}(flux, phi, grad.result) : Divergence{S}(flux, phi)

function bench(S, kind)
    if kind == :vector
        eqn = (Time{SteadyState}(U) + div_term(S, mdotf, U, gradU) - Laplacian{Linear}(nueff, U) ==
            Source(VectorField(mesh))) → VectorEquation(U, ())
        phi = U
    else
        eqn = (Time{SteadyState}(k) + div_term(S, mdotf, k, gradk) - Laplacian{Linear}(nueff, k) ==
            Source(ScalarField(mesh))) → ScalarEquation(k, ())
        phi = k
    end
    t_first = @elapsed discretise!(eqn, phi, config) # includes compilation
    discretise!(eqn, phi, config)
    alloc = @allocated discretise!(eqn, phi, config)
    ts = [@elapsed(discretise!(eqn, phi, config)) for _ in 1:reps]
    return t_first, minimum(ts), median(ts), alloc
end

schemes = [:Upwind, :Linear, :LUST, :BoundedUpwind, :LinearUpwind, :LimitedLinearUpwind]
@printf("# %s, %d cells, %d thread(s), julia %s\n", basename(mesh_file), ncells, Threads.nthreads(), VERSION)
@printf("%-8s %-20s %12s %12s %12s %10s\n", "eqn", "scheme", "first [ms]", "min [ms]", "median [ms]", "alloc [B]")
for kind in (:vector, :scalar), s in schemes
    isdefined(XCALibre, s) || continue
    S = getfield(XCALibre, s)
    tf, tmin, tmed, alloc = bench(S, kind)
    @printf("%-8s %-20s %12.1f %12.3f %12.3f %10d\n", kind, s, 1e3tf, 1e3tmin, 1e3tmed, alloc)
end
