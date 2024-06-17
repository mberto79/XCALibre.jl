# using Plots
using FVM_1D
using Krylov
using CUDA
using KernelAbstractions

using Accessors
using LinearAlgebra
using Statistics
using Krylov
using LinearOperators
using ProgressMeter
using Printf
using CUDA
using KernelAbstractions
using Atomix
using Adapt

# BFS_U CASE
mesh_file = "unv_sample_meshes/backwardFacingStep_10mm.unv"
mesh = build_mesh(mesh_file, scale=0.001)
# mesh = update_mesh_format(mesh, integer = Int32, float = Float32)
mesh = update_mesh_format(mesh)

velocity = [0.5, 0.0, 0.0]
nu = 1e-3
Re = velocity[1]*0.1/nu

model = RANS{Laminar}(mesh=mesh, viscosity=ConstantScalar(nu))

@assign! model U (
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Dirichlet(:wall, [0.0, 0.0, 0.0]),
    Dirichlet(:top, [0.0, 0.0, 0.0])
)

 @assign! model p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:wall, 0.0),
    Neumann(:top, 0.0)
)

schemes = (
    U = set_schemes(time=Euler),
    p = set_schemes()
)


solvers = (
    U = set_solver(
        model.U;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 1.0,
    ),
    p = set_solver(
        model.p;
        solver      = CgSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 1.0,
    )
)

runtime = set_runtime(
    iterations=1000, time_step=0.005, write_interval=10)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime)

GC.gc()

initialise!(model.U, velocity)
initialise!(model.p, 0.0)

@info "Extracting configuration and input fields..."
model = adapt(backend, model)
(; U, p, nu, mesh) = model
(; solvers, schemes, runtime) = config

@info "Preallocating fields..."

∇p = Grad{schemes.p.gradient}(p)
mdotf = FaceScalarField(mesh)
# nuf = ConstantScalar(nu) # Implement constant field!
rDf = FaceScalarField(mesh)
nueff = FaceScalarField(mesh)
initialise!(rDf, 1.0)
divHv = ScalarField(mesh)

@info "Defining models..."

ux_eqn = (
    Time{schemes.U.time}(U.x)
    + Divergence{schemes.U.divergence}(mdotf, U.x) 
    - Laplacian{schemes.U.laplacian}(nueff, U.x) 
    == 
    -Source(∇p.result.x)
) → Equation(mesh)

uy_eqn = (
    Time{schemes.U.time}(U.y)
    + Divergence{schemes.U.divergence}(mdotf, U.y) 
    - Laplacian{schemes.U.laplacian}(nueff, U.y) 
    == 
    -Source(∇p.result.y)
) → Equation(mesh)

uz_eqn = (
    Time{schemes.U.time}(U.z)
    + Divergence{schemes.U.divergence}(mdotf, U.z) 
    - Laplacian{schemes.U.laplacian}(nueff, U.z) 
    == 
    -Source(∇p.result.z)
) → Equation(mesh)

p_eqn = (
    Laplacian{schemes.p.laplacian}(rDf, p) == Source(divHv)
) → Equation(mesh)

# CUDA.allowscalar(false)
# model = _convert_array!(model, backend)
# ∇p = _convert_array!(∇p, backend)
# ux_eqn = _convert_array!(ux_eqn, backend)
# uy_eqn = _convert_array!(uy_eqn, backend)
# p_eqn = _convert_array!(p_eqn, backend)

@info "Initialising preconditioners..."

@reset ux_eqn.preconditioner = set_preconditioner(
                solvers.U.preconditioner, ux_eqn, U.x.BCs, runtime)
@reset uy_eqn.preconditioner = ux_eqn.preconditioner
@reset uz_eqn.preconditioner = ux_eqn.preconditioner
@reset p_eqn.preconditioner = set_preconditioner(
                solvers.p.preconditioner, p_eqn, p.BCs, runtime)

if isturbulent(model)
    @info "Initialising turbulence model..."
    turbulence = initialise_RANS(mdotf, p_eqn, config, model)
    config = turbulence.config
else
    turbulence = nothing
end

@info "Pre-allocating solvers..."
 
@reset ux_eqn.solver = solvers.U.solver(_A(ux_eqn), _b(ux_eqn))
@reset uy_eqn.solver = solvers.U.solver(_A(uy_eqn), _b(uy_eqn))
@reset uz_eqn.solver = solvers.U.solver(_A(uz_eqn), _b(uz_eqn))
@reset p_eqn.solver = solvers.p.solver(_A(p_eqn), _b(p_eqn))

 # Extract model variables and configuration
 (;mesh, U, p, nu) = model
 # ux_model, uy_model = ux_eqn.model, uy_eqn.model
 p_model = p_eqn.model
 (; solvers, schemes, runtime) = config
 (; iterations, write_interval) = runtime
 
 mdotf = get_flux(ux_eqn, 2)
 nueff = get_flux(ux_eqn, 3)
 rDf = get_flux(p_eqn, 1)
 divHv = get_source(p_eqn, 1)
 
 @info "Allocating working memory..."

 # Define aux fields 
 gradU = Grad{schemes.U.gradient}(U)
 gradUT = T(gradU)
 S = StrainRate(gradU, gradUT)
 S2 = ScalarField(mesh)

 # Temp sources to test GradUT explicit source
 # divUTx = zeros(Float64, length(mesh.cells))
 # divUTy = zeros(Float64, length(mesh.cells))

 n_cells = length(mesh.cells)
 Uf = FaceVectorField(mesh)
 pf = FaceScalarField(mesh)
 gradpf = FaceVectorField(mesh)
 Hv = VectorField(mesh)
 rD = ScalarField(mesh)

 # Pre-allocate auxiliary variables

 # Consider using allocate from KernelAbstractions 
 # e.g. allocate(backend, Float32, res, res)
 TF = _get_float(mesh)
 prev = zeros(TF, n_cells)
 prev = _convert_array!(prev, backend) 

 # Pre-allocate vectors to hold residuals 

 R_ux = ones(TF, iterations)
 R_uy = ones(TF, iterations)
 R_uz = ones(TF, iterations)
 R_p = ones(TF, iterations)
 
 interpolate!(Uf, U)   
 correct_boundaries!(Uf, U, U.BCs)
 flux!(mdotf, Uf)
 grad!(∇p, pf, p, p.BCs)

 update_nueff!(nueff, nu, turbulence)
 
 @info "Staring SIMPLE loops..."

#  progress = Progress(iterations; dt=1.0, showspeed=true)

#  CUDA.@tinzval_cpu = 2.2844029e-8
pref = nothing
# iteration = 1
for iteration in 1:1000
    @. prev = U.x.values
    discretise!(ux_eqn, prev, runtime)
    apply_boundary_conditions!(ux_eqn, U.x.BCs)
    implicit_relaxation!(ux_eqn, prev, solvers.U.relax, mesh)
    update_preconditioner!(ux_eqn.preconditioner, mesh)
    solve_system!(ux_eqn, solvers.U, U.x) #opP=Pu.P, solver=solver_U)
    residual!(R_ux, ux_eqn.equation, U.x, iteration)

    @. prev = U.y.values
    discretise!(uy_eqn, prev, runtime)
    apply_boundary_conditions!(uy_eqn, U.y.BCs)
    # uy_eqn.b .-= divUTy
    implicit_relaxation!(uy_eqn, prev, solvers.U.relax, mesh)
    update_preconditioner!(uy_eqn.preconditioner, mesh)
    solve_system!(uy_eqn, solvers.U, U.y)
    residual!(R_uy, uy_eqn.equation, U.y, iteration)

    inverse_diagonal!(rD, ux_eqn)
    interpolate!(rDf, rD)
    remove_pressure_source!(ux_eqn, uy_eqn, uz_eqn, ∇p)
    H!(Hv, U, ux_eqn, uy_eqn, uz_eqn)

    interpolate!(Uf, Hv)
    correct_boundaries!(Uf, Hv, U.BCs)
    div!(divHv, Uf)

    @. prev = p.values
    discretise!(p_eqn, prev, runtime)
    apply_boundary_conditions!(p_eqn, p.BCs)
    setReference!(p_eqn, pref, 1)
    update_preconditioner!(p_eqn.preconditioner, mesh)
    solve_system!(p_eqn, solvers.p, p)

    explicit_relaxation!(p, prev, solvers.p.relax)
    residual!(R_p, p_eqn.equation, p, iteration)

    grad!(∇p, pf, p, p.BCs) 

    correct_velocity!(U, Hv, ∇p, rD)
    interpolate!(Uf, U)
    correct_boundaries!(Uf, U, U.BCs)
    flux!(mdotf, Uf)
end

nzval_cpu = ux_eqn.equation.A.nzval
b_cpu = ux_eqn.equation.b
precon_cpu = ux_eqn.preconditioner.storage
values_cpu = U.x.values
R_ux_cpu = R_ux
rD_values_cpu = rD.values
rDf_values_cpu = rDf.values
Hx_cpu = Hv.x.values
Hy_cpu = Hv.y.values
Hz_cpu = Hv.z.values
Ufx_cpu = Uf.x.values
Ufy_cpu = Uf.y.values
Ufz_cpu = Uf.z.values
divHv_values_cpu = divHv.values
p_values_cpu = p.values 
mdotf_cpu = mdotf.values

nzvaly_cpu = uy_eqn.equation.A.nzval
by_cpu = uy_eqn.equation.b
precony_cpu = uy_eqn.preconditioner.storage
valuesy_cpu = U.y.values
R_uy_cpu = R_uy

nzvalp_cpu = p_eqn.equation.A.nzval
bp_cpu = p_eqn.equation.b
preconp_cpu = p_eqn.preconditioner.storage
valuesp_cpu = p.values
R_p_cpu = R_p
∇p_resultx_cpu = ∇p.result.x.values
∇p_resulty_cpu = ∇p.result.y.values
∇p_resultz_cpu = ∇p.result.z.values

function error_check(arr_cpu, arr_gpu, min_error)
    # CUDA.allowscalar(true)
    
    sum = 0

    error_array = eltype(arr_cpu)[]
    
    for i in eachindex(arr_cpu)

        varcpu = arr_cpu[i]
        vargpu = arr_gpu[i]

        diff = varcpu - vargpu
        
        if abs(diff) > min_error
            # println("Index  = $i:\nCPU Value = $varcpu\nGPU Value = $vargpu\nDifference = $diff\n")
            sum += 1
            push!(error_array, abs(diff))
        end
    end

    if length(error_array) > 0
        max_error = maximum(error_array)
        println("max error = $max_error")
        min_error = minimum(error_array)
        println("min error = $min_error")
    end

    println("number errored = $sum")
    # CUDA.allowscalar(false)
end

error_check(nzval_cpu, ux_eqn.equation.A.nzVal, eps(Float64))
error_check(b_cpu, ux_eqn.equation.b, eps(Float64))
error_check(ux_eqn.equation.A.nzVal, ux_eqn.preconditioner.A.nzVal, eps(Float64))
error_check(precon_cpu, ux_eqn.preconditioner.storage, eps(Float64))
error_check(values_cpu, U.x.values, eps(Float64))
error_check(R_ux_cpu, R_ux, eps(Float64))
error_check(rD_values_cpu, rD.values, eps(Float64))
error_check(rDf_values_cpu, rDf.values, eps(Float64))
error_check(Hx_cpu, Hv.x.values, eps(Float64))
error_check(Hy_cpu, Hv.y.values, eps(Float64))
error_check(Hz_cpu, Hv.z.values, eps(Float64))
error_check(Ufx_cpu, Uf.x.values, eps(Float64))
error_check(Ufy_cpu, Uf.y.values, eps(Float64))
error_check(Ufz_cpu, Uf.z.values, eps(Float64))
error_check(divHv_values_cpu, divHv.values, eps(Float64))
error_check(mdotf_cpu, mdotf.values, eps(Float64))

error_check(nzvaly_cpu, uy_eqn.equation.A.nzVal, eps(Float64))
error_check(by_cpu, uy_eqn.equation.b, eps(Float64))
error_check(uy_eqn.equation.A.nzVal, uy_eqn.preconditioner.A.nzVal, eps(Float64))
error_check(precony_cpu, uy_eqn.preconditioner.storage, eps(Float64))
error_check(valuesy_cpu, U.y.values, eps(Float64))
error_check(R_uy_cpu, R_uy, eps(Float64))

error_check(nzvalp_cpu, p_eqn.equation.A.nzVal, eps(Float64))
error_check(bp_cpu, p_eqn.equation.b, eps(Float64))
error_check(p_eqn.equation.A.nzVal, p_eqn.preconditioner.A.nzVal, eps(Float64))
error_check(preconp_cpu, p_eqn.preconditioner.storage, eps(Float64))
error_check(valuesp_cpu, p.values, eps(Float64))
error_check(R_p_cpu, R_p, eps(Float64))
error_check(∇p_resultx_cpu, ∇p.result.x.values, eps(Float64))
error_check(∇p_resulty_cpu, ∇p.result.y.values, eps(Float64))
error_check(∇p_resultz_cpu, ∇p.result.z.values, eps(Float64))

using KernelAbstractions
backend = CPU()
backend = CUDABackend()
using LinearAlgebra
function check_face_duplicates(mesh)
    for (fi, face) ∈ enumerate(mesh.faces)
        face = mesh.faces[fi] # face to check
        for i ∈ eachindex(mesh.faces)
            facei = mesh.faces[i]
            centre_diff = norm(face.centre - facei.centre)
            if centre_diff <= 1e-16
                # println("Face ", fi, " and face ", i, " share same location")
                if fi !== i 
                    println("Problem here!")
                end
            end
        end
    end
end

function size_cell_faces_array(mesh)
    nfaces = 0
    for cell ∈ mesh.cells
        nfaces += length(cell.faces_range)
    end
    nfaces == length(mesh.cell_faces) ? println("Pass: cell_faces ok") : println("FAIL")
    nothing
end

function check_boundary_normals(mesh)
    (; cells, faces, boundaries) = mesh
    for boundary ∈ boundaries
        for fID ∈ boundary.IDs_range
            face = faces[fID]
            own1 = face.ownerCells[1]
            own2 = face.ownerCells[2]
            if own1 !== own2
                println("Fail: Boundary faces can only have one owner")
            end
            cell = cells[own1]
            e = face.centre - cell.centre
            check = signbit(e ⋅ face.normal)
            if check
                println("Fail: face normal not correctly aligned")
            end
        end
    end
end

function check_internal_face_normals(mesh)
    (; cells, faces, cell_faces, cell_nsign) = mesh
    nfails = 0
    for cell ∈ cells
        for fi ∈ cell.faces_range
            fID = cell_faces[fi]
            nsign = cell_nsign[fi]
            face = faces[fID]
            e = face.centre - cell.centre
            check = signbit(e ⋅ face.normal)
            if (check && nsign !== -1) || (!check && nsign !== 1)
                nfails += 1
                println("Fail: Normal not consistent on ", nfails, " faces")
            end
        end
    end
end

@time check_face_duplicates(mesh)
@time size_cell_faces_array(mesh)
@time check_boundary_normals(mesh)
@time check_internal_face_normals(mesh)