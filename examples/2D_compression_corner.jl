# using Plots
using XCALibre
# using CUDA # Run this if using NVIDIA GPU
# using AMDGPU # Run this if using AMD GPU

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "compression_corner_2d_32_64_SF4.unv"

mesh_file = joinpath(grids_dir, grid)
mesh = UNV2D_mesh(mesh_file, scale=0.001)


backend = CPU(); workgroup = 1024
# backend = CUDABackend(); workgroup = 32 
# backend = ROCBackend(); workgroup = 32


mesh_dev = adapt(backend, mesh)

# Inlet conditions

# Not working

nu = 0.0 # 1e-10
gamma = 1.4
cp = 1005.0
R = cp*(1.0 - (1.0/gamma))
cv = cp - R
temp = 1000 # 300.0
Tref= 0 # 298.15
pressure = 100000
Pr = 0.7
h = cp*temp + Tref

M = 2
a = sqrt(gamma*R*temp)
rho = pressure/(R*temp)# p = rho R T 
Umag = M*a

velocity = [Umag, 0.0, 0.0]
noSlip = [0.0, 0.0, 0.0]
Re = (0.2*velocity[1])/nu

model = Physics(
    time = Steady(),
    # time = Transient(),
    fluid = Fluid{Compressible}(
    # fluid = Fluid{WeaklyCompressible}(
        nu = nu,
        cp = cp,
        gamma = gamma,
        Pr = Pr
        ),
    turbulence = RANS{Laminar}(),
    energy = Energy{SensibleEnthalpy}(Tref=Tref),
    # energy = Energy{InternalEnergy}(Tref=Tref),
    domain = mesh_dev
    )

boundaries = assign(
    region = mesh,
    (
        U = [
            Dirichlet(:inlet, velocity),
            # Zerogradient(:inlet),
            Zerogradient(:outlet),
            Zerogradient(:top),
            # Symmetry(:top),
            # Slip(:top),
            # Extrapolated(:outlet),
            # Wall(:wall, noSlip)
            # Symmetry(:wall)
            Slip(:wall)
        ],
        p = [
            # Zerogradient(:inlet),
            # Dirichlet(:outlet, pressure),
            Dirichlet(:inlet, pressure),
            Zerogradient(:outlet),

            Zerogradient(:top),
            # Symmetry(:top),
            # Slip(:top),

            # Wall(:wall)
            # Slip(:wall)
            Zerogradient(:wall)
        ],
        he = [
            FixedTemperature(:inlet, T=temp, Enthalpy(cp=cp, Tref=Tref)),
            # FixedTemperature(:inlet, T=temp, IE(cv=cv, Tref=Tref)),
            # Zerogradient(:outlet),
            Zerogradient(:outlet),
            # Extrapolated(:outlet),
            Zerogradient(:top),
            # Symmetry(:top),
            # Slip(:top),

            # Wall(:wall)
            # Slip(:wall)
            Zerogradient(:wall)
            # FixedTemperature(:wall, T=temp, Enthalpy(cp=cp, Tref=Tref))
        ]
    )
)

atol = 1e-2
rtol = 0.0
solvers = (
    U = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-10,
        relax       = 0.7,
        # relax       = 1.0,
        rtol = rtol,
        atol = atol
    ),
    p = SolverSetup(
        # solver      = Cg(), # Bicgstab(), Gmres(), Cg() # WeaklyCompressible
        solver      = Bicgstab(), # Bicgstab(), Gmres(), Cg() # Compressible
        preconditioner = Jacobi(), # Jacobi DILU
        convergence = 1e-10,
        relax       = 0.3,
        # relax       = 1.0,
        limit = (0.5*pressure, 5*pressure),
        rtol = rtol,
        atol = atol
    ),
    he = SolverSetup(
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-10,
        relax       = 0.3,
        # relax       = 1.0,
        limit = (800, 3000.0),
        rtol = rtol,
        atol = atol
    )
)

time = SteadyState # SteadyState Euler
divergence = BoundedUpwind # Upwind BoundedUpwind LUST
divergence_p = Upwind
limiter = nothing # CellBased()
schemes = (
    U = Schemes(time=time, divergence=divergence, limiter=limiter),
    p = Schemes(time=time,divergence=divergence_p),
    he = Schemes(time=time, divergence=divergence, limiter=limiter)
)

runtime = Runtime(iterations=10000, write_interval=100, time_step=1)
# runtime = Runtime(iterations=10000, write_interval=100, time_step=1e-5)

hardware = Hardware(backend=backend, workgroup=workgroup)

config = Configuration(;
    solvers, schemes, runtime, hardware, boundaries)

GC.gc(true)

# initialise!(model.momentum.U, [0.0, 0.0, 0.0])
initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, pressure)
initialise!(model.energy.T, temp)

residuals = run!(model, config, output=VTK()); #, pref=0.0)



# using Plots
# using DelimitedFiles

# # Load single file and inspect data and record headers/columns
# M_case = "2"
# # M_case = "2.5"
# data, header = open("compression_15deg_M$M_case.csv", "r") do io 
#         readdlm(io, ',', header=true)
# end

# header[1] # ux
# header[2] # uy
# header[3] # uz
# header[4] # p
# header[5] # rho 
# header[6] # T
# header[7] # mask
# header[8] # line

# xdir = data[:,8]

# # M = 2
# p_ratio = 2.19465313
# t_ratio = 1.26937635
# rho_ratio = 1.72892233
# M2 =  1.44571634

# #  M = 2.5
# # p_ratio =  2.46750012
# # t_ratio =  1.32195866
# # rho_ratio =  1.86654863
# # M2 =   1.87352598


# Mach = @. sqrt(data[:,1]^2 + data[:,2]^2 + 0*data[:,3]^2)/sqrt(gamma*R*data[:,6])


# default(
#     linewidth=2, color=:black, label=nothing, 
#     fg_legend = :false, legend=:bottomleft,
#     legendfont=6, titlefont=10)
# p1 = plot(xdir, data[:,6], title="Temperature", color=:red)
# plot!(p1, [xdir[1], xdir[end]], [temp, temp], label="Analytical")
# plot!(p1, [xdir[1], xdir[end]], [temp*t_ratio, temp*t_ratio])

# p2 = plot(xdir, data[:,5], title="Density", color=:red)
# plot!(p2, [xdir[1], xdir[end]], [rho, rho], label="Analytical")
# plot!(p2, [xdir[1], xdir[end]], [rho*rho_ratio, rho*rho_ratio])

# p3 = plot(xdir, data[:,4], title="Pressure", color=:red)
# plot!(p3, [xdir[1], xdir[end]], [pressure, pressure], label="Analytical")
# plot!(p3, [xdir[1], xdir[end]], [pressure*p_ratio, pressure*p_ratio])

# p4 = plot(xdir, Mach, title="Mach", color=:red)
# plot!(p4, [xdir[1], xdir[end]], [M, M], label="Analytical")
# plot!(p4, [xdir[1], xdir[end]], [M2, M2])

# plot(p1,p2,p3,p4) |> display