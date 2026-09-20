#= Assemble the SAME equation with both assemblies and compare the matrix entry by entry.
No solver, no iteration: any difference here is a bug, not amplification. =#
using XCALibre, Printf, LinearAlgebra, Random

mesh = UNV2D_mesh(joinpath(pkgdir(XCALibre), "examples/0_GRIDS", "backwardFacingStep_10mm.unv"), scale=0.001)
model = Physics(time=Steady(), solid=Solid{Uniform}(k=1.0), energy=Energy{Conduction}(), domain=mesh)
BCs = assign(region = mesh, (
    T = [Dirichlet(:inlet, 50.0), Zerogradient(:outlet), Dirichlet(:wall, 10.0), Zerogradient(:top)],))

phi   = ScalarField(mesh)
gamma = FaceScalarField(mesh)
mdotf = FaceScalarField(mesh)
src   = ScalarField(mesh)
Random.seed!(1234)
phi.values   .= rand(length(phi.values))
gamma.values .= 0.5 .+ rand(length(gamma.values))
mdotf.values .= randn(length(mdotf.values))   # both signs, so upwinding is exercised
src.values   .= rand(length(src.values))

cfg(asm) = Configuration(
    solvers = (T = SolverSetup(solver=Cg(), preconditioner=Jacobi(), convergence=1e-8, relax=1.0),),
    schemes = (T = Schemes(time=SteadyState, divergence=Upwind, laplacian=Linear, gradient=Gauss),),
    runtime = Runtime(iterations=1, write_interval=-1, time_step=1),
    hardware = Hardware(backend=CPU(), workgroup=64, assembly=asm),
    boundaries = BCs)

function build(asm, scheme_div)
    eqn = (
        Divergence{scheme_div}(mdotf, phi) - Laplacian{Linear}(gamma, phi) == Source(src)
        ) → ScalarEquation(phi, BCs.T)
    discretise!(eqn, phi, cfg(asm))
    A = eqn.equation.A
    copy(XCALibre.ModelFramework._nzval(A)), copy(eqn.equation.b)
end

println("cells = ", length(mesh.cells), "  faces = ", length(mesh.faces))
for div in (Upwind, Linear, LUST)
    nzc, bc = build(CellAssembly(), div)
    nzf, bf = build(FaceAssembly(), div)
    dnz = maximum(abs.(nzc .- nzf)); snz = maximum(abs.(nzc))
    db  = maximum(abs.(bc .- bf));   sb  = maximum(abs.(bc))
    nbad = count(abs.(nzc .- nzf) .> 1e-10*snz)
    @printf("%-8s  max|dA|=%.3e (rel %.3e)  entries differing=%d/%d   max|db|=%.3e (rel %.3e)\n",
            div, dnz, dnz/snz, nbad, length(nzc), db, db/max(sb,eps()))
end
