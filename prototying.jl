using Plots

using FVM_1D

using Krylov
using LinearOperators
using ILUZero
using LoopVectorization


# backwardFacingStep_2mm, backwardFacingStep_10mm
mesh_file = "unv_sample_meshes/backwardFacingStep_10mm.unv"
mesh = build_mesh(mesh_file, scale=0.001)

# struct Mesh2{I,F} <: AbstractMesh
#     cells::Vector{Cell{I,F}}
#     faces::Vector{Face2D{I,F}}
#     boundaries::Vector{Boundary{I}}
#     nodes::Vector{Node{F}}
# end

# mesh = Mesh2(mesh.cells, mesh.faces, (mesh.boundaries...), mesh.nodes)

p = ScalarField(mesh)
U = VectorField(mesh)

velocity = [0.5, 0.0, 0.0]
nu = 1e-3
Re = velocity[1]*0.1/nu

UBCs = (
    Dirichlet(U, :inlet, velocity),
    Neumann(U, :outlet, 0.0),
    Dirichlet(U, :wall, [0.0, 0.0, 0.0]),
    Dirichlet(U, :top, [0.0, 0.0, 0.0])
    # Neumann(U, :top, 0.0)
    )

uxBCs = (
    Dirichlet(U, :inlet, velocity[1]),
    Neumann(U, :outlet, 0.0),
    Dirichlet(U, :wall, 0.0),
    Dirichlet(U, :top, 0.0)
    # Neumann(:top, 0.0)
)

uyBCs = (
    Dirichlet(U, :inlet, velocity[2]),
    Neumann(U, :outlet, 0.0),
    Dirichlet(U, :wall, 0.0),
    Dirichlet(U, :top, 0.0)
    # Neumann(:top, 0.0)
)

pBCs = (
    Neumann(p, :inlet, 0.0),
    Dirichlet(p, :outlet, 0.0),
    Neumann(p, :wall, 0.0),
    Neumann(p, :top, 0.0)
)

setup_U = SolverSetup(
    solver      = BicgstabSolver,
    relax       = 0.8,
    itmax       = 100,
    rtol        = 1e-1
)

setup_p = SolverSetup(
    solver      = GmresSolver, #CgSolver, #GmresSolver, #BicgstabSolver,
    relax       = 0.2,
    itmax       = 100,
    rtol        = 1e-2
)

using Profile, PProf

GC.gc()

p = ScalarField(mesh)
U = VectorField(mesh)


# Pre-allocate fields
ux = ScalarField(mesh)
uy = ScalarField(mesh)
∇p = Grad{Linear}(p)
mdotf = FaceScalarField(mesh)
nuf = ConstantScalar(nu) # Implement constant field! Priority 1
rDf = FaceScalarField(mesh)
divHv_new = ScalarField(mesh)


# Define models 
model_ux = (
    Divergence{Linear}(mdotf, ux) - Laplacian{Linear}(nuf, ux) 
    == 
    Source(∇p.x)
)

model_uy = (
    Divergence{Linear}(mdotf, uy) - Laplacian{Linear}(nuf, uy) 
    == 
    Source(∇p.y)
)

model_p = (
    Laplacian{Linear}(rDf, p) == Source(divHv_new)
)

# Extract model variables
ux = model_ux.terms[1].phi
mdotf = model_ux.terms[1].flux
uy = model_uy.terms[1].phi
nuf = model_ux.terms[2].flux
rDf = model_p.terms[1].flux 
rDf.values .= 1.0
divHv_new = ScalarField(model_p.sources[1].field, mesh)

# Define aux fields 
n_cells = m = n = length(mesh.cells)

# U = VectorField(mesh)
Uf = FaceVectorField(mesh)
# mdot = ScalarField(mesh)

pf = FaceScalarField(mesh)
# ∇p = Grad{Midpoint}(p)
gradpf = FaceVectorField(mesh)

Hv = VectorField(mesh)
Hvf = FaceVectorField(mesh)
Hv_flux = FaceScalarField(mesh)
divHv = Div(Hv, FaceVectorField(mesh), zeros(Float64, n_cells), mesh)
rD = ScalarField(mesh)

# Define equations
ux_eqn  = Equation(mesh)
uy_eqn  = Equation(mesh)
p_eqn    = Equation(mesh)

# Pre-allocated auxiliary variables
ux0 = zeros(Float64, n_cells)
uy0 = zeros(Float64, n_cells)
p0 = zeros(Float64, n_cells)

ux0 .= velocity[1]
uy0 .= velocity[2]
p0 .= zero(Float64)

# Define preconditioners and linear operators
opAx = LinearOperator(ux_eqn.A)
Px = ilu0(ux_eqn.A)
opPUx = LinearOperator(Float64, m, n, false, false, (y, v) -> ldiv!(y, Px, v))

opAy = LinearOperator(uy_eqn.A)
Py = ilu0(uy_eqn.A)
opPUy = LinearOperator(Float64, m, n, false, false, (y, v) -> ldiv!(y, Py, v))

discretise!(p_eqn, model_p)
apply_boundary_conditions!(p_eqn, model_p, pBCs)
opAp = LinearOperator(p_eqn.A)
opPP = opLDL(p_eqn.A)

solver_p = setup_p.solver(p_eqn.A, p_eqn.b)
solver_U = setup_U.solver(ux_eqn.A, ux_eqn.b)

#### NEED TO IMPLEMENT A SENSIBLE INITIALISATION TO INCLUDE WARM START!!!!
# Update initial (guessed) fields

@turbo ux0 .= ux.values
@turbo uy0 .= uy.values 
@turbo p0 .= p.values
@inbounds ux.values .= velocity[1]
@inbounds uy.values .= velocity[2]
@turbo U.x .= ux.values #velocity[1]
@turbo U.y .= uy.values# velocity[2]
# end
volumes(mesh) = [mesh.cells[i].volume for i ∈ eachindex(mesh.cells)]
volume  = volumes(mesh)
rvolume  = 1.0./volume

interpolate!(Uf, U)   
correct_boundaries!(Uf, U, UBCs)
flux!(mdotf, Uf)

source!(∇p, pf, p, pBCs)

# R_ux = TF[]
# R_uy = TF[]
# R_p = TF[]
iterations = 200
R_ux = ones(Float64, iterations)
R_uy = ones(Float64, iterations)
R_p = ones(Float64, iterations)

# Perform SIMPLE loops 
# progress = Progress(iterations; dt=1.0, showspeed=true)
# @time for iteration ∈ 1:iterations
    iteration = 1
    
    source!(∇p, pf, p, pBCs)
    Solvers.neg!(∇p)

    discretise!(ux_eqn, model_ux)
    @turbo @. uy_eqn.A.nzval = ux_eqn.A.nzval
    @time apply_boundary_conditions!(ux_eqn, model_ux, uxBCs)
    @time Solvers.implicit_relaxation!(ux_eqn, ux0, setup_U.relax)
    A = ux_eqn.A
    @time ilu0!(Px, A)
    @time run!( # 6 allocs
        ux_eqn, model_ux, uxBCs, 
        setup_U, opA=opAx, opP=opPUx, solver=solver_U
    )
    @time Solvers.residual!(R_ux, ux_eqn, ux, opAx, solver_U, iteration)# 2 allocs


    # print("Solving Uy...")

    @turbo @. uy_eqn.b = 0.0
    # discretise!(uy_eqn, model_uy)
    apply_boundary_conditions!(uy_eqn, model_uy, uyBCs)
    Solvers.implicit_relaxation!(uy_eqn, uy0, setup_U.relax)
    ilu0!(Py, uy_eqn.A)
    run!(
        uy_eqn, model_uy, uyBCs, 
        setup_U, opA=opAy, opP=opPUy, solver=solver_U
    )
    Solvers.residual!(R_uy, uy_eqn, uy, opAy, solver_U, iteration)


    @turbo for i ∈ eachindex(ux0)
        ux0[i] = U.x[i]
        uy0[i] = U.y[i]
        # U.x[i] = ux.values[i]
        # U.y[i] = uy.values[i]
    end
    
    Solvers.inverse_diagonal!(rD, ux_eqn)
    interpolate!(rDf, rD)
    Solvers.remove_pressure_source!(ux_eqn, uy_eqn, ∇p, rD)
    # H!(Hv, U, ux_eqn, uy_eqn)
    Solvers.H!(Hv, ux, uy, ux_eqn, uy_eqn, rD)
    
    # @turbo for i ∈ eachindex(ux0)
    #     U.x[i] = ux0[i]
    #     U.y[i] = uy0[i]
    # end

    # div!(divHv, UBCs) # 7 allocations
    # @turbo @. divHv.values *= rvolume
    
    interpolate!(Hvf, Hv)
    correct_boundaries!(Hvf, Hv, UBCs)
    Solvers.flux!(Hv_flux, Hvf)
    div!(divHv_new, Hv_flux)
    # @turbo @. divHv_new.values *= rvolume

    # @inbounds @. rD.values *= volume
    # interpolate!(rDf, rD)
    # @inbounds @. rD.values *= rvolume

    # print("Solving p...")

    
    discretise!(p_eqn, model_p)
    apply_boundary_conditions!(p_eqn, model_p, pBCs)
    Solvers.setReference!(p_eqn, nothing, 1)
    @time run!( # 36 allocs
        p_eqn, model_p, pBCs, 
        setup_p, opA=opAp, opP=opPP, solver=solver_p
    )

    @time grad!(∇p, pf, p, pBCs) 
    correct = false
    if correct
        ncorrectors = 1
        for i ∈ 1:ncorrectors
            discretise!(p_eqn, model_p)
            apply_boundary_conditions!(p_eqn, model_p, pBCs)
            setReference!(p_eqn, pref, 1)
            # grad!(∇p, pf, p, pBCs) 
            interpolate!(gradpf, ∇p, p)
            nonorthogonal_flux!(pf, gradpf) # careful: using pf for flux (not interpolation)
            correct!(p_eqn, model_p.terms.term1, pf)
            run!(
                p_eqn, model_p, pBCs, 
                setup_p, opA=opAp, opP=opPP, solver=solver_p
            )
            grad!(∇p, pf, p, pBCs) 
        end
    end

    # source!(∇p, pf, p, pBCs)
    
    correct_velocity!(U, Hv, ∇p, rD)
    interpolate!(Uf, U)
    correct_boundaries!(Uf, U, UBCs)
    flux!(mdotf, Uf)

    
    explicit_relaxation!(p, p0, setup_p.relax)
    residual!(R_p, p_eqn, p, opAp, solver_p, iteration)

    # source!(∇p, pf, p, pBCs)
    grad!(∇p, pf, p, pBCs) 
    correct_velocity!(ux, uy, Hv, ∇p, rD)

    # push!(R_ux, r_ux)
    # push!(R_uy, r_uy)
    # push!(R_p, r_p)
    # R_ux[iteration] = r_ux
    # R_uy[iteration] = r_uy
    # R_p[iteration] = r_p
    convergence = 1e-7
    if R_ux[iteration] <= convergence && R_uy[iteration] <= convergence && R_p[iteration] <= convergence
        print("\nSimulation converged! ($iteration iterations)\n")
        break

iterations = 1000
Rx, Ry, Rp = isimple!(
    mesh, velocity, nu, U, p, 
    uxBCs, uyBCs, pBCs, UBCs,
    # setup_U, setup_p, iterations, pref=0.0)
    setup_U, setup_p, iterations)

write_vtk("results", mesh, ("U", U), ("p", p))

plot(; xlims=(0,230))
plot!(1:length(Rx), Rx, yscale=:log10)
plot!(1:length(Ry), Ry, yscale=:log10)
plot!(1:length(Rp), Rp, yscale=:log10)