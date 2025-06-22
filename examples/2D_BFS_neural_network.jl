using Plots
using XCALibre
using Flux
using StaticArrays
using Statistics
using LinearAlgebra
using KernelAbstractions
# using CUDA

actual(y) = begin
    H = 1 # channel height
    H2 = H/2
    h = y - H2
    vx = (1 - (h/H2)^2)
    return vx
end

inflowNetwork = Chain(
    Dense(1 => 6, sigmoid),
    Dense(6 => 1)) |> f64

y_actual = [0:0.01:1;] # array of y-values for plotting
vx_actual = actual.(y_actual)

# Generate training dataset
y_train = hcat(rand(0:(0.1/100):0.1, 100)...)./0.1
vx_train = actual.(y_train)

# Test locations selected randomly
y_test = hcat(rand(0:(0.1/100):0.1, 100)...)./0.1
vx_untrained = inflowNetwork(y_test)

plot(
    y_actual, vx_actual, label="Actual", 
    frame_style=:box, foreground_color_legend = nothing,
    xlabel="Dimensionless distance", ylabel="Normalised velocity")
scatter!(y_train', vx_train', label="Training data")
scatter!(y_test', vx_untrained', label="Untrained output")


loss(inflowNetwork, y, vx) = mean(abs2.(inflowNetwork(y) .- vx))

opt =  Flux.setup(Adam(), inflowNetwork)
data = [(y_train, vx_train)]
for epoch in 1:20000
    Flux.train!(loss, inflowNetwork, data, opt)
end
loss(inflowNetwork, data[1]...,)

vx_trained = inflowNetwork(y_test)


plot(
    y_actual, vx_actual, label="Actual", 
    frame_style=:box, foreground_color_legend = nothing,
    xlabel="Dimensionless distance", ylabel="Normalised velocity")
scatter!(y_test', vx_trained', label="Trained output")


struct Inflow{F,I,O,N,V,T} <: XCALibreUserFunctor
    U::F        # maximum velocity
    H::F        # inlet height
    input::I    # vector to hold input coordinates
    output::O   # vector to hold model inferred values
    network::N  # model itself
    xdir::V     # struct used to define x-direction unit vector
    steady::T   # required field! (Bool)
end
Adapt.@adapt_structure Inflow

(bc::Inflow)(vec, t, i) = begin
    velocity = @view bc.output[:,i]
    return @inbounds SVector{3}(velocity[1], velocity[2], velocity[3])
end

# import XCALibre.Discretise: update_boundary!

XCALibre.Discretise.update_user_boundary!(
    BC::DirichletFunction{I,V,R}, faces, cells, facesID_range, time, config
    ) where{I,V<:Inflow,R} = 
begin

    (; hardware) = config
    (; backend, workgroup) = hardware

    kernel_range = length(facesID_range)
    kernel! = _update_user_boundary!(backend, workgroup, kernel_range)
    kernel!(BC, faces, cells, facesID_range, time, ndrange=kernel_range)
    KernelAbstractions.synchronize(backend)

    (; output, input, U, network, xdir) = BC.value
    output .= U.*network(input).*xdir # convert to vector
end

@kernel function _update_user_boundary!(BC, faces, cells, facesID_range, time)
    i = @index(Global)
    startID = facesID_range[1]
    fID = i + startID - 1
    coords = faces[fID].centre
    BC.value.input[i] = coords[2]/BC.value.H # scale coordinates
end


grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "backwardFacingStep_5mm.unv"
mesh_file = joinpath(grids_dir, grid)

mesh = UNV2D_mesh(mesh_file, scale=0.001)

nfaces = mesh.boundaries[1].IDs_range |> length

# backend = CUDABackend(); workgroup = 32
backend = CPU(); workgroup = 1024; activate_multithread(backend)

hardware = set_hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

U = 0.5 # maximum velocity
H = 0.1 # inlet height
input = zeros(1,nfaces)
input .= (H/2)/H
output = U.*inflowNetwork(input).*[1 0 0]'
@view output[:,2]

inlet_profile= Inflow(
    0.5,
    0.1,
    input,
    output,
    inflowNetwork,
    [1,0,0],
    true
)

velocity = [0.5, 0.0, 0.0]
nu = 1e-3
Re = velocity[1]*0.1/nu

model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh
    )

BCs = assign(
    region=mesh_dev,
    (
        U = [
            DirichletFunction(:inlet, inlet_profile), # Pass functor
            Extrapolated(:outlet),
            Wall(:wall, [0.0, 0.0, 0.0]),
            Wall(:top, [0.0, 0.0, 0.0])
        ],
        p = [
            Extrapolated(:inlet),
            Dirichlet(:outlet, 0.0),
            Wall(:wall),
            Wall(:top)
        ]
    )
)

schemes = (
    U = set_schemes(divergence = Linear),
    p = set_schemes()
)


solvers = (
    U = set_solver(
        region = mesh_dev,
        solver      = Bicgstab(),
        preconditioner = Jacobi(),
        convergence = 1e-8,
        relax       = 0.7,
        rtol = 1e-4
    ),
    p = set_solver(
        region = mesh_dev,
        solver      = Cg(),
        preconditioner = Jacobi(),
        convergence = 1e-8,
        relax       = 0.3,
        rtol = 1e-4
    )
)

runtime = set_runtime(iterations=500, time_step=1, write_interval=500)
# runtime = set_runtime(iterations=1, time_step=1, write_interval=-1) # hide

hardware = set_hardware(backend=CPU(), workgroup=1024)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

GC.gc()

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

residuals = run!(model, config)
