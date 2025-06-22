# Verification: 2D Unsteady incompressible cylinder

# Introduction
---

To verify the correct implementation of the laminar flow solver, a simulation of a transient, laminar and incompressible cylinder was carried out and compared with OpenFOAM. The results show that the solver in XCALibre.jl generates similar results compared to the laminar solver in OpenFOAM. Simulation set up, grid and OpenFOAM case will all be made available here.

## Results

![vorticity comparison with OpenFOAM](figures/02/cylinder_re100_comparison.gif)

## Simulation setup

For those interested in running this case, this simulation can be replicated as follows.

```jldoctest;  filter = r".*"s => s"", output = false

using XCALibre
# using CUDA # uncomment to run on GPU

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "cylinder_d10mm_5mm.unv"
mesh_file = joinpath(grids_dir, grid)

mesh = UNV2D_mesh(mesh_file, scale=0.001)

mesh_dev = mesh
# mesh_dev = adapt(CUDABackend(), mesh) # uncomment to run on GPU

# Inlet conditions
velocity = [0.5, 0.0, 0.0]
noSlip = [0.0, 0.0, 0.0]
nu = 1e-3
Re = (0.2*velocity[1])/nu

model = Physics(
    time = Transient(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
    )

BCs = assign(
    region=mesh_dev,
    (
        U = [
                Dirichlet(:inlet, velocity),
                Extrapolated(:outlet),
                Wall(:cylinder, noSlip),
                Extrapolated(:bottom),
                Extrapolated(:top)
        ],
        p = [
                Extrapolated(:inlet),
                Dirichlet(:outlet, 0.0),
                Wall(:cylinder),
                Extrapolated(:bottom),
                Extrapolated(:top)
        ]
    )
)

solvers = (
    U = set_solver(
        region = mesh_dev,
        solver      = Bicgstab(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 1.0,
        rtol = 1e-4,
        atol = 1e-5
    ),
    p = set_solver(
        region = mesh_dev,
        solver      = Cg(), # Bicgstab(), Gmres()
        preconditioner = Jacobi(), #NormDiagonal(),
        convergence = 1e-7,
        relax       = 0.8,
        rtol = 1e-4,
        atol = 1e-5
    )
)

schemes = (
    U = set_schemes(time=Euler, divergence=LUST, gradient=Gauss),
    p = set_schemes(time=Euler, gradient=Gauss)
)


runtime = set_runtime(iterations=1000, write_interval=50, time_step=0.005) 
runtime = set_runtime(iterations=1, write_interval=-1, time_step=0.005) # hide


hardware = set_hardware(backend=CPU(), workgroup=1024)
# hardware = set_hardware(backend=CUDABackend(), workgroup=32) # uncomment to run on GPU

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

GC.gc(true)

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

residuals = run!(model, config)

# output

```