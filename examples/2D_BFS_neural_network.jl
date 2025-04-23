using Plots
using XCALibre
using Flux
using StaticArrays
using Statistics
using LinearAlgebra
using KernelAbstractions
using Adapt
# using CUDA

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
loss(inflowNetwork, y, vx) = mean(abs2.(inflowNetwork(y) .- vx))

# opt = Descent(0.3)
opt =  Flux.setup(Adam(), inflowNetwork)
data = [(y_train, vx_train)]
for epoch in 1:20000
    Flux.train!(loss, inflowNetwork, data, opt)
end
loss(inflowNetwork, data[1]...,)

scatter(y_train', inflowNetwork(y_train)', legend=:none)
scatter!(vec(y_train), actual.(vec(y_train)), legend=:none)

# inflowNetwork([1])
# inflowNetwork_dev = inflowNetwork |> gpu
# inflowNetwork_dev(SVector{1}(1))


struct Inflow{F,I,O,N,V,B} <: XCALibreUserFunctor
    U::F
    H::F
    input::I
    output::O
    network::N
    xdir::V
    steady::B
end
Adapt.@adapt_structure Inflow

# import XCALibre.Discretise: update_boundary!

XCALibre.Discretise.update_user_boundary!(
    BC::DirichletFunction{I,V}, eqnModel, component, faces, cells, facesID_range, time, config) where{I,V<:Inflow}= begin
    # if time > 1 # for this to work need to add time to steady solvers! # to do
    #     return nothing
    # end

    (; hardware) = config
    (; backend, workgroup) = hardware

    kernel_range = length(facesID_range)
    kernel! = _update_user_boundary!(backend, workgroup, kernel_range)
    kernel!(BC, eqnModel, component, faces, cells, facesID_range, time, ndrange=kernel_range)
    # KernelAbstractions.synchronize(backend)
    BC.value.output .= BC.value.network(BC.value.input).*BC.value.xdir
end

@kernel function _update_user_boundary!(BC, eqnModel, component, faces, cells, facesID_range, time)
    i = @index(Global)
    startID = facesID_range[1]
    fID = i + startID - 1
    coords = faces[fID].centre
    BC.value.input[i] = coords[2]/BC.value.H
end


grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "backwardFacingStep_5mm.unv"
mesh_file = joinpath(grids_dir, grid)

mesh = UNV2D_mesh(mesh_file, scale=0.001)

nfaces = mesh.boundaries[1].IDs_range |> length
U = 0.5
H = 0.1
input = zeros(1,nfaces)
input .= (H/2)/H
output = U.*inflowNetwork(input).*[1 0 0]'
@view output[:,2]

inlet= Inflow(
    0.5,
    0.1,
    input,
    output,
    inflowNetwork,
    [1,0,0], 
    true
)

# test = adapt(CUDABackend(), Storage([0.0], [0.0,0.0,0.0]))
inlet_dev = inlet
# inlet_dev = adapt(CUDABackend(), inlet)

(bc::Inflow)(vec, t, i) = begin
    velocity = @view bc.output[:,i]
    return @inbounds SVector{3}(velocity[1], velocity[2], velocity[3])
    # return @inbounds SVector{3}(velocity)
end

# inlet.update!(inlet, (0,0.0,0),0,2)


res = inlet(SVector{3}(0,0.05,0),0,2)
res = inlet_dev(SVector{3}(0,0.05,0),0,2)
CUDA.@allowscalar res = inlet_dev(SVector{3}(0,0.05,0),0,2)



# mesh_dev = adapt(CUDABackend(), mesh)
mesh_dev = mesh

velocity = [0.5, 0.0, 0.0]
nu = 1e-3
Re = velocity[1]*0.1/nu

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
    )

@assign! model momentum U (
    DirichletFunction(:inlet, inlet_dev),
    # DirichletFunction(:inlet, inlet),
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
    iterations=1000, time_step=1, write_interval=100)

# hardware = set_hardware(backend=CUDABackend(), workgroup=32)
hardware = set_hardware(backend=CPU(), workgroup=1024)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

GC.gc()

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

residuals = run!(model, config)# 9.39k allocs in 184 iterations
