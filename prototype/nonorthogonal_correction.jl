using Plots
using XCALibre
using Accessors
using Statistics

# quad and trig 40 and 100
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "trig.unv"
grid = "trig40.unv"
# grid = "trig100.unv"
# grid = "quad.unv"
# grid = "quad40.unv"
# grid = "quad100.unv"

mesh_file = joinpath(grids_dir, grid)
mesh = UNV2D_mesh(mesh_file, scale=0.001)

backend = CPU()

phi = ScalarField(mesh)
phif = FaceScalarField(mesh)
q = ScalarField(mesh)
gammaf = FaceScalarField(mesh)
# gradScheme = Orthogonal
gradScheme = Midpoint
∇phi = Grad{gradScheme}(phi)

# Diffusion solver

eqn = (
        - Laplacian{Linear}(gammaf, phi) == Source(q)
    ) → ScalarEquation(mesh)

phi = assign(
    phi, 
    # Dirichlet(:inlet, 0.0),
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    # Dirichlet(:bottom, 0.0),
    Neumann(:bottom, 0.0),
    Dirichlet(:top, 2.0),
    )


solvers= (; 
    phi = set_solver(
        phi;
        solver      = CgSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(), 
        convergence = 1e-7,
        relax       = 0.8,
        rtol = 0.01
    )
)

schemes = (;phi = set_schemes(divergence=Upwind, gradient=Midpoint))
runtime = set_runtime(iterations=500, write_interval=100, time_step=1)
hardware = set_hardware(backend=backend, workgroup=1024)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

@reset eqn.preconditioner = set_preconditioner(
    solvers.phi.preconditioner, eqn, phi.BCs, config)
@reset eqn.solver = solvers.phi.solver(_A(eqn), _b(eqn))

gammaf.values .= 1
prev = zeros(length(phi.values))
initialise!(phi, 1)

itmax = 300
ncorrectors = 1
for iteration ∈ 1:itmax

    prev .= phi.values
    discretise!(eqn, phi, config)   
    apply_boundary_conditions!(eqn, phi.BCs, nothing, time, config)
    update_preconditioner!(eqn.preconditioner, phi.mesh, config)
    solve_system!(eqn, config.solvers.phi, phi, nothing, config)
    explicit_relaxation!(phi, prev, solvers.phi.relax, config)
    # residual = mean(sqrt.((_b(eqn) - _A(eqn)*phi.values).^2))
    
    grad!(∇phi, phif, phi, phi.BCs, 0.0, config)

    # non-orthogonal correction
    for i ∈ 1:ncorrectors
        # prev .= phi.values
        discretise!(eqn, phi, config)   
        apply_boundary_conditions!(eqn, phi.BCs, nothing, time, config)
        XCALibre.Solvers.nonorthogonal_face_correction(eqn, ∇phi, gammaf, config)
        update_preconditioner!(eqn.preconditioner, phi.mesh, config)
        solve_system!(eqn, config.solvers.phi, phi, nothing, config)
        explicit_relaxation!(phi, prev, solvers.phi.relax, config)
        grad!(∇phi, phif, phi, phi.BCs, 0.0, config)
        # limit_gradient!(limit_gradient, ∇p, p, config)
    end

    # explicit_relaxation!(phi, prev, solvers.phi.relax, config)
    residual = mean(sqrt.((_b(eqn) - _A(eqn)*phi.values).^2))
    
    println("Iteration $iteration, residual: $residual")
    if residual < 1e-4
        println("Converged in $iteration iterations.")
        break
    end
end


# grad!(∇phi, phif, phi, phi.BCs, 0.0, config) 
# limit_gradient!(∇phi, phif, phi, config)

meshData = VTKWriter2D(nothing, nothing)
write_results("output", mesh, meshData, ("phi", phi), ("gradPhi", ∇phi.result))
# write_results("output_qud", mesh, meshData, ("phi", phi), ("gradPhi", ∇phi.result))
