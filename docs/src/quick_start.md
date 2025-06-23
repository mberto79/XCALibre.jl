# Quick Start

*Read this section for information about how to install XCALibre.jl and an example showcasing the API*

## Installation
---

First, you need to [download and install Julia on your system](https://julialang.org/downloads/). Once you have a working installation of Julia, XCALibre.jl can be installed using the built-in package manager. 

XCALibre.jl is available directly from the the General Julia Registry. Thus, to install XCALibre.jl open a Julia REPL, press `]` to enter the package manager. The REPL prompt icon will change from **julia>** (green) to **pkg>** (and change colour to blue) or **(myenvironment) pkg>** where `myenvironment` is the name of the currently active Julia environment. Once you have activated the package manager mode enter

```julia
pkg> add XCALibre
```

To install XCALibre.jl directly from Github enter the following command (for the latest release)

```julia
pkg> add https://github.com/mberto79/XCALibre.jl.git
```

A specific branch can be installed by providing the branch name precided by a `#`, for example, to install the `dev-0.3-main` branch enter

```julia
pkg> add https://github.com/mberto79/XCALibre.jl.git#dev-0.3-main
```

!!! note
    
    To enable GPU acceleration you will also need to install the corresponding GPU package for your hardware. See CUDA.jl, AMD.jl, oneAPI.jl for more details. XCALibre.jl will automatically precompile and load the relevant backend specific functionality (using [Julia extensions](https://pkgdocs.julialang.org/v1/creating-packages/#Conditional-loading-of-code-in-packages-(Extensions)))

## Example
---

The example below illustrates the top-level API used in XCALibre.jl. It shows the key steps a user needs to follow to set up a simulation:

* [Pre-processing](@ref) (steps 1 and 2)
* [Physics and models](@ref) (steps 3 to 5)
* [Numerical setup](@ref) (steps 6 and 7)
* [Runtime and solvers](@ref) setup (steps 8 to 11)
* [Post-processing](@ref) (step 12)

Once you have installed Julia and XCALibre.jl, the example below can be run by copying the contents shown below and pasting them in a file. The file can be executed within vscode, the Julia REPL or from a system terminal. 

* To run in vscode check the information for using [Julia in vscode](https://code.visualstudio.com/docs/languages/julia)
* To run in the [Julia REPL](https://docs.julialang.org/en/v1/stdlib/REPL/), simply launch Julia and type `include("name_of_your_file.jl")`
* To run from a system terminal (bash or cmd, for example), simply type `path/to/julia name_of_your_file.jl`

In most cases, it is preferable to run simulations from within the Julia REPL because, in Julia, there is often a cost associated to the first run due to compilation time. By relaunching a simulation in the REPL, all previously compiled code will not be recompiled. This is particularly helpful in the prototyping stages. For long running simulations, the compilation time is normally negligible compared to the actual time needed to complete the simulation.

```jldoctest;  filter = r".*"s => s"", output = false

# Step 0. Load libraries
using XCALibre
# using CUDA # Uncomment to run on NVIDIA GPUs
# using AMDGPU # Uncomment to run on AMD GPUs


# Step 1. Define mesh
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "backwardFacingStep_10mm.unv"
mesh_file = joinpath(grids_dir, grid)

mesh = UNV2D_mesh(mesh_file, scale=0.001) # convert mesh

# Step 2. Select backend and setup hardware
backend = CPU()
# backend = CUDABackend() # ru non NVIDIA GPUs
# backend = ROCBackend() # run on AMD GPUs

hardware = set_hardware(backend=backend, workgroup=1024)
# hardware = set_hardware(backend=backend, workgroup=32) # use for GPU backends

mesh_dev = mesh # use this line to run on CPU
# mesh_dev = adapt(backend, mesh)  # Uncomment to run on GPU 

# Step 3. Flow conditions
velocity = [1.5, 0.0, 0.0]
nu = 1e-3
Re = velocity[1]*0.1/nu

# Step 4. Define physics
model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
    )

# Step 5. Define boundary conditions
BCs = assign(
    region=mesh_dev,
    (
        U = [
            Dirichlet(:inlet, velocity),
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

# Step 6. Choose discretisation schemes
schemes = (
    U = set_schemes(divergence = Linear),
    p = set_schemes() # no input provided (will use defaults)
)

# Step 7. Set up linear solvers and preconditioners
solvers = (
    U = set_solver(
        region = mesh_dev,
        solver      = Bicgstab(), # Options: Gmres()
        preconditioner = Jacobi(), # Options: NormDiagonal()
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 1e-1
    ),
    p = set_solver(
        region = mesh_dev,
        solver      = Cg(), # Options: Cg(), Bicgstab(), Gmres()
        preconditioner = Jacobi(), # Options: NormDiagonal()
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 1e-2
    )
)

# Step 8. Specify runtime requirements
runtime = set_runtime(iterations=2000, time_step=1, write_interval=2000)
runtime = set_runtime(iterations=1, time_step=1, write_interval=-1) # hide

# Step 9. Construct Configuration object
config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

# Step 10. Initialise fields (initial guess)
initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)

# Step 11. Run simulation
residuals = run!(model, config);

# Step 12. Post-process
pwd() # find active directory where the file "iteration_002000.vtk" was saved

# output

```

### Output

If you chose to run the example above, XCALibre.jl would have written a simulation result file to your computer. The name of the file is `iteration_002000.vtk`. The file will be located in your current active directory (you can check this by running `pwd()`). This file can be open directly in ParaView for further post-processing. You can find out more about [ParaView on their website](https://www.paraview.org/). The image below is the output solution generated by XCALibre.jl to the example simulation above.

![Simulation result visualisation in ParaView](figures/quick_start_fig_bfs_2d_incompressible_laminar.svg)
