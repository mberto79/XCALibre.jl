#= Same matrix comparison, on the motorBike mesh with the real BC set (Wall/Slip extend the
sparsity) and the real k-equation term list. =#
using XCALibre, JLD2, Printf, Random

mesh = load_object("/home/humberto/casesXCALibre/XCALibre_benchmarks/3D_motorBike_RANS/XCALibre/mesh.jld2")
nb = length(mesh.boundary_cellsID)
@printf("cells=%d faces=%d boundary_cellsID=%d  sum(patch sizes)=%d\n",
    length(mesh.cells), length(mesh.faces), nb,
    sum(length(b.IDs_range) for b in mesh.boundaries))

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

phi = ScalarField(mesh); mueff = FaceScalarField(mesh); mdotf = FaceScalarField(mesh)
Dkf = ScalarField(mesh); Pk = ScalarField(mesh); rho = ConstantScalar(1.0)
Random.seed!(7)
phi.values   .= rand(length(phi.values))
mueff.values .= 1e-5 .+ rand(length(mueff.values))
mdotf.values .= randn(length(mdotf.values))
Dkf.values   .= rand(length(Dkf.values))
Pk.values    .= rand(length(Pk.values))

cfg(asm) = Configuration(
    solvers = (k = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-8, relax=1.0),),
    schemes = (k = Schemes(time=SteadyState, divergence=Upwind, laplacian=Linear, gradient=Gauss),),
    runtime = Runtime(iterations=1, write_interval=-1, time_step=1),
    hardware = Hardware(backend=CPU(static=true), workgroup=AutoTune(), assembly=asm),
    boundaries = BCs)

function build(asm)
    eqn = (
        Time{SteadyState}(rho, phi)
        + Divergence{Upwind}(mdotf, phi)
        - Laplacian{Linear}(mueff, phi)
        + Si(Dkf, phi)
        == Source(Pk)
        ) → ScalarEquation(phi, BCs.k)
    discretise!(eqn, phi, cfg(asm))
    copy(XCALibre.ModelFramework._nzval(eqn.equation.A)), copy(eqn.equation.b)
end

nzc, bc = build(CellAssembly())
nzf, bf = build(FaceAssembly())
d = abs.(nzc .- nzf); s = maximum(abs.(nzc))
@printf("threads=%d  nnz=%d\n", Threads.nthreads(), length(nzc))
@printf("max|dA| = %.6e  (rel %.3e)   entries differing > 1e-12*scale = %d\n",
        maximum(d), maximum(d)/s, count(d .> 1e-12*s))
@printf("max|db| = %.6e\n", maximum(abs.(bc .- bf)))
if maximum(d) > 0
    idx = findall(d .> 1e-12*s)
    @printf("first 5 differing nz indices: %s\n", string(idx[1:min(5,end)]))
    for i in idx[1:min(5,end)]
        @printf("   nz=%d  cell=%.17g  face=%.17g\n", i, nzc[i], nzf[i])
    end
end
