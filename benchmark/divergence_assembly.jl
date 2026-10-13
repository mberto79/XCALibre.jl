# Assembly benchmark of the convection term (minimum of `reps` warmed `discretise!` calls per scheme).
# Usage: julia --project=<tree> [-t N] benchmark/divergence_assembly.jl [mesh.unv] [reps]
# Run on a baseline and a branch checkout and compare; schemes a checkout lacks are skipped.
using XCALibre, Statistics, Printf
import KernelAbstractions as KA
const USE_CUDA = get(ENV, "BENCH_CUDA", "0") == "1" # needs CUDA in the environment
USE_CUDA && @eval using CUDA

mesh_file = length(ARGS) >= 1 && !isempty(ARGS[1]) ? ARGS[1] :
    pkgdir(XCALibre, "examples/0_GRIDS/backwardFacingStep_2mm.unv")
reps = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 50

mesh = UNV2D_mesh(mesh_file, scale=0.001)
backend = USE_CUDA ? Base.invokelatest(() -> CUDA.CUDABackend()) : CPU()
hardware = Hardware(backend=backend, workgroup=USE_CUDA ? 32 : 1024)
mesh = adapt(backend, mesh)
ncells = length(mesh.cells)
config = Configuration(
    solvers=nothing, schemes=nothing, boundaries=nothing, hardware=hardware,
    runtime=Runtime(iterations=1, write_interval=-1, time_step=1))

mdotf = FaceScalarField(mesh)
copyto!(mdotf.values, 1e-3 .* (1 .+ 0.1 .* rand(length(mdotf.values))))
U = VectorField(mesh); copyto!(U.x.values, rand(ncells))
k = ScalarField(mesh); copyto!(k.values, rand(ncells))
gradU = Grad{Gauss}(U)
gradk = Grad{Gauss}(k)
nueff = FaceScalarField(mesh); fill!(nueff.values, 1e-3)

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
    run() = (discretise!(eqn, phi, config); KA.synchronize(backend))
    t_first = @elapsed run() # includes compilation
    run()
    alloc = @allocated run()
    ts = [@elapsed(run()) for _ in 1:reps]
    return t_first, minimum(ts), median(ts), alloc
end

schemes = [:Upwind, :Linear, :LUST, :BoundedUpwind, :LinearUpwind, :LimitedLinearUpwind]
@printf("# %s, %d cells, %s, %d thread(s), julia %s\n", basename(mesh_file), ncells, USE_CUDA ? "CUDA" : "CPU", Threads.nthreads(), VERSION)
@printf("%-8s %-20s %12s %12s %12s %10s\n", "eqn", "scheme", "first [ms]", "min [ms]", "median [ms]", "alloc [B]")
for kind in (:vector, :scalar), s in schemes
    isdefined(XCALibre, s) || continue
    S = getfield(XCALibre, s)
    tf, tmin, tmed, alloc = bench(S, kind)
    @printf("%-8s %-20s %12.1f %12.3f %12.3f %10d\n", kind, s, 1e3tf, 1e3tmin, 1e3tmed, alloc)
end
