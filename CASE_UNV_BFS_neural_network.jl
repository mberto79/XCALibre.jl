using Plots
using FVM_1D
using Flux
using Statistics
using LinearAlgebra
using CUDA
using StaticArrays

actual(y) = begin
    H = 1 # channel height
    H2 = H/2
    h = y - H2
    vx = (1 - (h/H2)^2)
    return vx
end

y_train = hcat(rand(0:(0.1/100):0.1, 100)...)./0.1
vx_train = actual.(y_train)

inflowNetwork = Chain(
    Dense(1 => 6, sigmoid),   # activation function inside layer
    # Dense(12 => 12, sigmoid),   # activation function inside layer
    Dense(6 => 1)) # |> gpu 


# loss(model, y, vx) = mean(abs2.(model(y) .- vx))
loss(y, vx) = mean(abs2.(inflowNetwork(y) .- vx))

# opt = Descent(0.3)
opt =  Flux.setup(Adam(), inflowNetwork)
data = [(y_train, vx_train)]
for epoch in 1:10000
    Flux.train!(loss, inflowNetwork, data, opt)
end
loss(data[1]...,)

scatter(y_train', inflowNetwork(y_train)', legend=:none)
scatter!(vec(y_train), actual.(vec(y_train)), legend=:none)

inflowNetwork_dev = inflowNetwork |> gpu

inflow(vec, t) = begin
    H = 0.1
    U = 0.5
    yhat = vec[2]/H
    # vxhat = inflowNetwork(SVector{1}(yhat))[]
    # vxhat = inflowNetwork_dev(SVector{1}(yhat))[]
    # velocity = inflowNetwork_dev([yhat])*[0,U,0]'
    # velocity = inflowNetwork([yhat])*SVector{3}(0,U,0)' # works CPU
    velocity = inflowNetwork_dev([yhat])[] # *SVector{3}(0,U,0)'
    # vx = U*vxhat
    velocity = SVector{3}(vx, 0, 0)
    return velocity
end

inflow([0,0.05,0],0)

# backwardFacingStep_2mm, backwardFacingStep_10mm
mesh_file = "unv_sample_meshes/backwardFacingStep_10mm.unv"
mesh = UNV2D_mesh(mesh_file, scale=0.001)

mesh_dev = adapt(CUDABackend(), mesh)
mesh_dev = mesh

velocity = [0.5, 0.0, 0.0]
nu = 1e-3
Re = velocity[1]*0.1/nu

model = Physics(
    time = Steady(),
    fluid = FLUID{Incompressible}(nu = nu),
    turbulence = RANS{Laminar}(),
    energy = ENERGY{Isothermal}(),
    domain = mesh_dev
    )

@assign! model momentum U (
    DirichletFunction(:inlet, inflow),
    # Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Dirichlet(:wall, [0.0, 0.0, 0.0]),
    Dirichlet(:top, [0.0, 0.0, 0.0]),
    # Wall(:wall, [0.0, 0.0, 0.0]),
    # Wall(:top, [0.0, 0.0, 0.0])
    # Symmetry(:top, 0.0)
)

@assign! model momentum p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:wall, 0.0),
    Neumann(:top, 0.0)
    # Symmetry(:top, 0.0)
)

schemes = (
    U = set_schemes(divergence = Linear),
    # U = set_schemes(divergence = Upwind),
    p = set_schemes()
)


solvers = (
    U = set_solver(
        model.momentum.U;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 1e-4,
        atol = 1e-10
    ),
    p = set_solver(
        model.momentum.p;
        solver      = CgSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.3,
        rtol = 1e-4,
        atol = 1e-10
    )
)

runtime = set_runtime(
    iterations=2000, time_step=1, write_interval=100)

hardware = set_hardware(backend=CUDABackend(), workgroup=32)
hardware = set_hardware(backend=CPU(), workgroup=32)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

GC.gc()

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

Rx, Ry, Rz, Rp, model_out = run!(model, config) # 9.39k allocs in 184 iterations

plot(; xlims=(0,1000))
plot!(1:length(Rx), Rx, yscale=:log10, label="Ux")
plot!(1:length(Ry), Ry, yscale=:log10, label="Uy")
plot!(1:length(Rp), Rp, yscale=:log10, label="p")

# # PROFILING CODE

using Profile, PProf

GC.gc()
initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

Profile.Allocs.clear()
Profile.Allocs.@profile sample_rate=1.0 begin 
    Rx, Ry, Rz, Rp, model_out = run!(model, config)
end

# Profile.print(format=:flat)

PProf.Allocs.pprof()

PProf.refresh()

@profview_allocs Rx, Ry, Rz, Rp, model_out = run!(model, config) sample_rate=1