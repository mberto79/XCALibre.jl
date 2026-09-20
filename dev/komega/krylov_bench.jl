#= Isolated Krylov cost on the real motorBike matrices. The end-to-end A/B could not resolve
   the BLAS-1 effect above +/-6% run-to-run noise, so this fixes the work (atol=rtol=0,
   itmax=ITMAX) and takes min-of-reps on the solve alone.
     julia --project=dev/komega -t N dev/komega/krylov_bench.jl [reps] [out]
=#
using XCALibre, JLD2, Printf, Random, Krylov, LinearOperators, LinearAlgebra
using SparseMatricesCSR, ThreadPinning

reps  = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 10
out   = length(ARGS) >= 2 ? ARGS[2] : "dev/komega/krylov_bench.txt"
ITMAX = 50

pinthreads(:cores)
mesh = load_object("/home/humberto/casesXCALibre/XCALibre_benchmarks/3D_motorBike_RANS/XCALibre/mesh.jld2")
velocity = [20.0,0.0,0.0]
model = Physics(time=Steady(), fluid=Fluid{Incompressible}(nu=1.5e-5),
    turbulence=RANS{KOmega}(), energy=Energy{Isothermal}(), domain=mesh)
BCs = assign(region = mesh, (
    U = [Dirichlet(:inlet, velocity), Zerogradient(:outlet), Wall(:lowerWall, velocity),
         Wall(:motorBike, [0.0,0.0,0.0]), Slip(:upperWall), Slip(:frontAndBack)],
    p = [Zerogradient(:inlet), Dirichlet(:outlet, 0.0), Wall(:lowerWall),
         Wall(:motorBike), Slip(:upperWall), Slip(:frontAndBack)],
    k = [Dirichlet(:inlet, 0.24), Zerogradient(:outlet), KWallFunction(:lowerWall),
         KWallFunction(:motorBike), Slip(:upperWall), Slip(:frontAndBack)],
    omega = [Dirichlet(:inlet, 1.78), Zerogradient(:outlet), OmegaWallFunction(:lowerWall),
         OmegaWallFunction(:motorBike), Slip(:upperWall), Slip(:frontAndBack)],
    nut = [Dirichlet(:inlet, 0.13), Zerogradient(:outlet), NutWallFunction(:lowerWall),
         NutWallFunction(:motorBike), Slip(:upperWall), Slip(:frontAndBack)]))

cfg = Configuration(
    solvers = (k = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-8, relax=1.0),),
    schemes = (k = Schemes(time=SteadyState, divergence=Upwind, laplacian=Linear, gradient=Gauss),),
    runtime = Runtime(iterations=1, write_interval=-1, time_step=1),
    hardware = Hardware(backend=CPU(static=true), workgroup=AutoTune()),
    boundaries = BCs)

Random.seed!(7)
phi = ScalarField(mesh); mueff = FaceScalarField(mesh); mdotf = FaceScalarField(mesh)
Dkf = ScalarField(mesh); Pk = ScalarField(mesh); rho = ConstantScalar(1.0)
phi.values .= rand(length(phi.values)); mueff.values .= 1e-5 .+ rand(length(mueff.values))
mdotf.values .= randn(length(mdotf.values)); Dkf.values .= rand(length(Dkf.values))
Pk.values .= rand(length(Pk.values))

# unsymmetric transport matrix (k/omega/U shape) and a symmetric Laplacian (p shape)
function build(sym)
    eqn = sym ?
        (- Laplacian{Linear}(mueff, phi) == Source(Pk)) → ScalarEquation(phi, BCs.p) :
        (Time{SteadyState}(rho, phi) + Divergence{Upwind}(mdotf, phi)
         - Laplacian{Linear}(mueff, phi) + Si(Dkf, phi) == Source(Pk)) → ScalarEquation(phi, BCs.k)
    discretise!(eqn, phi, cfg)
    apply_boundary_conditions!(eqn, sym ? BCs.p : BCs.k, nothing, nothing, cfg)
    eqn.equation
end
eqA = build(false); eqS = build(true)
A_un, b_un = eqA.A, eqA.b
A_sy, b_sy = eqS.A, eqS.b
n = length(b_un)
nzv(A) = XCALibre.ModelFramework._nzval(A)
rpt(A) = XCALibre.ModelFramework._rowptr(A)
cvl(A) = XCALibre.ModelFramework._colval(A)

# Int32 copy of the addressing; nzval stays Float64
function to_i32(A)
    P = parent(A)
    SparseXCSR(SparseMatricesCSR.SparseMatrixCSR{1}(
        size(P,1), size(P,2), Vector{Int32}(rpt(A)), Vector{Int32}(cvl(A)), copy(nzv(A))))
end
A_un32 = to_i32(A_un); A_sy32 = to_i32(A_sy)

# Jacobi diagonal, exactly as update_Jacobi! builds it
function jac(A)
    d = zeros(Float64, n)
    rp = rpt(A); cv = cvl(A); nz = nzv(A)
    for i in 1:n, j in rp[i]:(rp[i+1]-1)
        cv[j] == i && (d[i] = 1/abs(nz[j]))
    end
    d
end
d_un = jac(A_un); d_sy = jac(A_sy)

# threaded replacement for opDiagonal: the LinearOperators broadcast is serial, and it runs
# once (Cg) or twice (Bicgstab) per Krylov iteration on a full-length vector
function thr_diag(d::Vector{T}) where T
    f = (res, v, α, β) -> begin
        if β == zero(T)
            Threads.@threads :static for i in eachindex(res); @inbounds res[i] = α*d[i]*v[i]; end
        else
            Threads.@threads :static for i in eachindex(res); @inbounds res[i] = α*d[i]*v[i] + β*res[i]; end
        end
        res
    end
    LinearOperator(T, length(d), length(d), true, true, f, f, f)
end

ws_un = _workspace(Bicgstab(), b_un)
ws_sy = _workspace(Cg(), b_sy)
x0 = zeros(n)

function timed(ws, A, b, M)
    fill!(x0, 0.0)
    t = @elapsed krylov_solve!(ws, A, b, x0; M=M, itmax=ITMAX, atol=0.0, rtol=0.0,
                               ldiv=false, history=false)
    t, Krylov.iteration_count(ws)
end

variants = [
    ("blas1 opDiag i64", 1, :op,  :i64),
    ("blas8 opDiag i64", 8, :op,  :i64),
    ("blas1 thrDiag i64",1, :thr, :i64),
    ("blas8 thrDiag i64",8, :thr, :i64),
    ("blas1 opDiag i32", 1, :op,  :i32),
    ("blas8 thrDiag i32",8, :thr, :i32),
]
best = Dict{String,NTuple{2,Float64}}(); iters = Dict{String,NTuple{2,Int}}()
for (name,_,_,_) in variants; best[name] = (Inf, Inf); end

for r in 1:reps, (name, nb, pre, ix) in variants
    BLAS.set_num_threads(nb)
    Au = ix === :i32 ? A_un32 : A_un; As = ix === :i32 ? A_sy32 : A_sy
    Mu = pre === :thr ? thr_diag(d_un) : opDiagonal(d_un)
    Ms = pre === :thr ? thr_diag(d_sy) : opDiagonal(d_sy)
    tu, iu = timed(ws_un, Au, b_un, Mu)
    ts, is = timed(ws_sy, As, b_sy, Ms)
    r == 1 && continue                      # first rep is warm-up/compile
    best[name] = (min(best[name][1], tu), min(best[name][2], ts))
    iters[name] = (iu, is)
end

# primitive breakdown at this size
x = rand(n); y = rand(n); z = zeros(n)
mn(f, k=200) = minimum(@elapsed(f()) for _ in 1:k)
prim = String[]
for nb in (1, 8)
    BLAS.set_num_threads(nb)
    push!(prim, @sprintf("blas=%d  spmv %.3f ms   opDiag %.3f   thrDiag %.3f   dot %.3f   axpy %.3f",
        nb,
        1000*mn(()->mul!(z, A_un, x)),
        1000*mn(()->mul!(z, opDiagonal(d_un), x, 1.0, 0.0)),
        1000*mn(()->mul!(z, thr_diag(d_un), x, 1.0, 0.0)),
        1000*mn(()->BLAS.dot(n, x, 1, y, 1)),
        1000*mn(()->BLAS.axpy!(1.0, x, y))))
end
BLAS.set_num_threads(1)
push!(prim, @sprintf("i32 spmv (blas irrelevant) %.3f ms", 1000*mn(()->mul!(z, A_un32, x))))

open(out, "w") do f
    @printf(f, "julia threads=%d  n=%d  nnz=%d  itmax=%d  reps=%d (first discarded)\n",
        Threads.nthreads(), n, length(nzv(A_un)), ITMAX, reps)
    @printf(f, "%-20s %12s %12s %10s %10s\n", "variant", "bicgstab ms", "cg ms", "bi_it", "cg_it")
    for (name,_,_,_) in variants
        @printf(f, "%-20s %12.2f %12.2f %10d %10d\n", name,
            1000*best[name][1], 1000*best[name][2], iters[name][1], iters[name][2])
    end
    println(f, "\nper Krylov iteration (ms):")
    for (name,_,_,_) in variants
        @printf(f, "%-20s bicgstab %7.4f   cg %7.4f\n", name,
            1000*best[name][1]/iters[name][1], 1000*best[name][2]/iters[name][2])
    end
    println(f, "\nprimitives:"); for p in prim; println(f, "  ", p); end
    println(f, "\nSIMPLE-iteration projection (motorBike: p 25.78 cg, U 3x8.5 bicgstab, k+omega 2.1 bicgstab)")
    for (name,_,_,_) in variants
        cgi = best[name][2]/iters[name][2]; bii = best[name][1]/iters[name][1]
        tot = 25.78*cgi + (3*8.5 + 2.1)*bii
        @printf(f, "%-20s %7.1f ms/SIMPLE-iter  -> %6.1f s over 500\n", name, 1000tot, 500tot)
    end
end
println("done")
