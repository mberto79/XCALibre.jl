# Advanced: 2D Aerofoil inflow optimisation

# Introduction
---

Here, a simple optimisation is performed on a 2D NACA0012 aerofoil case. The optimal angle of attack for maximum lift-to-drag ratio is found using `BayesianOptimization.jl`. This example serves to illustrate both how `XCALibre.jl` can easily integrate with the Julia ecosystem, and the ease with which post-processing functions can be written.

# Optimisation Setup
---

For those interested in running this example, the optimisation can be replicated by following these steps.

### Install and load modules

To be able to run this example the following modules need to be installed. This can be done by entering into package mode (using "]" in the REPL) and typing the following:

```julia
add Plots, Distributions, LinearAlgebra, GaussianProcesses, BayesianOptimization
```

This will download and install the required packages. Note that "CUDA" must also be added for GPU acceleration. Once installed, the packages can be loaded as follows:

```julia
using Pkg; # hide
installed = "Flux" ∈ keys(Pkg.project().dependencies) # hide
installed && Pkg.rm("Flux", io=devnull) #hide
Pkg.add("BayesianOptimization", io=devnull) # hide

using XCALibre, BayesianOptimization, Plots
using LinearAlgebra, GaussianProcesses, Distributions
# using CUDA # uncomment to run on GPU
"done"
nothing # hide
```

### Define post processing functions

In order to calculate the lift-to-drag ratio, the pressure and viscous forces over the aerofoil must be calculated and summed. The lift and drag components can then be calculated (noting the rotated inflow vector to allow for different aerofoil angles of attack with the same mesh). A function returning the coefficients of lift and drag (not specifically needed for this example) is also presented for convenience.

```julia
# Lift to drag ratio calculation
lift_to_drag(patch::Symbol, model, ρ, nu, α) = begin
    Fp = pressure_force(patch, model.momentum.p, ρ)
    Fv = viscous_force(patch, model.momentum.U, ρ, nu, model.turbulence.nut)
    Ft = Fp + Fv
    Ft = [cos(-α*π/180) -sin(-α*π/180) 0; sin(-α*π/180) cos(-α*π/180) 0; 0 0 1]*Ft # Rotation matrix to account for rotated inflow
    aero_eff = Ft[2]/Ft[1]
    print("Aerofoil L/D: ",round(aero_eff,sigdigits = 4))
    return aero_eff
end 

# Aerodynamic coefficient calculation
aero_coeffs(patch::Symbol, chord, velocity, model, ρ, nu, α) = begin
    Fp = pressure_force(patch, model.momentum.p, ρ)
    Fv = viscous_force(patch, model.momentum.U, ρ, nu, model.turbulence.nut)
    Ft = Fp + Fv
    Ft = [cos(-α*π/180) -sin(-α*π/180) 0; sin(-α*π/180) cos(-α*π/180) 0; 0 0 1]*Ft # Rotation matrix to account for rotated inflow
    C_l = 2Ft[2]/(ρ*(velocity[1]^2)*chord*0.001)
    C_d = 2Ft[1]/(ρ*(velocity[1]^2)*chord*0.001)
    print("Lift Coefficient: ",round(C_l,sigdigits = 4))
    print("\nDrag Coefficient: ",round(C_d,sigdigits = 4))
    return C_l,C_d
end 
```

### Import 2D aerofoil mesh

Next, the aerofoil mesh .unv file must be imported. It must also be adapted to work with the GPU, if desired.

```julia
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "NACAMesh.unv"
mesh_file = joinpath(grids_dir, grid)

mesh = UNV2D_mesh(mesh_file, scale=0.001)
mesh_dev = mesh # running on CPU 
# mesh_dev = adapt(CUDABackend(), mesh) # uncomment to run on GPU
```

### Setup the CFD simulation as a function to be optimised

The `BayesianOptimization.jl` package can optimise a Julia function that is passed to it. To interface with XCALibre.jl, the entire CFD simulation setup must simply be wrapped within a function, which can then be passed to the optimiser. The function must take the variable to be changed (angle of attack, in this case) as its input, and must return the desired output (lift-to-drag ratio). Therefore, the post-processing step to calculate lift-to-drag ratio is also wrapped in the same function.

```julia
function foil_optim(α::Vector{Float64})
    println("\nSelected α value: $(α[1])")

    # Parameters
    chord = 250.0
    Re = 500000
    nu, ρ = 1.48e-5, 1.225
    Umag = (Re*nu)/(chord*0.001) # Calculate velocity magnitude for given Reynolds number
    velocity = [Umag*cos(α[1]*π/180), Umag*sin(α[1]*π/180), 0.0] # Velocity calculation
    νR = 10
    Tu = 0.025
    k_inlet = 3/2*(Tu*norm(velocity))^2
    ω_inlet = k_inlet/(νR*nu)

    # Boundary Conditions
    noSlip = [0.0, 0.0, 0.0]

    model = Physics(
        time = Steady(),
        fluid = Fluid{Incompressible}(nu = nu),
        turbulence = RANS{KOmega}(),
        energy = Energy{Isothermal}(),
        domain = mesh_dev
        )

    @assign! model momentum U ( 
        XCALibre.Dirichlet(:inlet, velocity),
        XCALibre.Dirichlet(:bottom, velocity),
        Neumann(:outlet, 0.0),
        Neumann(:top, 0.0),
        Wall(:foil, noSlip)
    )

    @assign! model momentum p (
        Neumann(:inlet, 0.0),
        Neumann(:bottom, 0.0),
        XCALibre.Dirichlet(:outlet, 0.0),
        XCALibre.Dirichlet(:top, 0.0),
        Neumann(:foil, 0.0)
    )

    @assign! model turbulence k (
        XCALibre.Dirichlet(:inlet, k_inlet),
        Neumann(:outlet, 0.0),
        Neumann(:top, 0.0),
        Neumann(:bottom, 0.0),
        XCALibre.Dirichlet(:foil, 1e-15)
    )

    @assign! model turbulence omega (
        XCALibre.Dirichlet(:inlet, ω_inlet),
        Neumann(:outlet, 0.0),
        Neumann(:top, 0.0),
        Neumann(:bottom, 0.0),
        OmegaWallFunction(:foil)
    )

    @assign! model turbulence nut (
        Neumann(:inlet, 0.0),
        Neumann(:outlet, 0.0),
        Neumann(:top, 0.0),
        Neumann(:bottom, 0.0), 
        XCALibre.Dirichlet(:foil, 0.0)
    )

    schemes = (
        U = set_schemes(divergence=Upwind, gradient=Midpoint),
        p = set_schemes(divergence=Upwind),
        k = set_schemes(divergence=Upwind, gradient=Midpoint),
        omega = set_schemes(divergence=Upwind, gradient=Midpoint)
    )

    solvers = (
        U = set_solver(
            model.momentum.U;
            solver = BicgstabSolver, # BicgstabSolver, GmresSolver
            preconditioner = Jacobi(), # Jacobi
            convergence = 1e-7,
            relax = 0.6,
            rtol = 1e-1,
        ),
        p = set_solver(
            model.momentum.p;
            solver = GmresSolver, # change to BicgstabSolver for GPU runs
            preconditioner = Jacobi(), # change to Jacobi() for GPU runs
            convergence = 1e-7,
            relax = 0.2,
            rtol = 1e-2,
        ),
        k = set_solver(
            model.turbulence.k;
            solver = BicgstabSolver,
            preconditioner = Jacobi(),
            convergence = 1e-7,
            relax = 0.6,
            rtol = 1e-1,
        ),
        omega = set_solver(
            model.turbulence.omega;
            solver      = BicgstabSolver,
            preconditioner = Jacobi(), 
            convergence = 1e-7,
            relax       = 0.6,
            rtol = 1e-1,
        )
    )

    runtime = set_runtime(iterations=500, write_interval=500, time_step=1)
    runtime = set_runtime(iterations=20, write_interval=-1, time_step=1) # hide

    hardware = set_hardware(backend=CPU(), workgroup=1024)
    # hardware = set_hardware(backend=CUDABackend(), workgroup=32) # uncomment to run on GPU

    config = Configuration(solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

    GC.gc()

    initialise!(model.momentum.U, velocity)
    initialise!(model.momentum.p, 0.0)
    initialise!(model.turbulence.k, k_inlet)
    initialise!(model.turbulence.omega, ω_inlet)
    initialise!(model.turbulence.nut, k_inlet/ω_inlet)

    residuals = run!(model, config)

    # Residuals Graph
    let
        iterations = 1:length(residuals.Ux)
        plot(; xlims=(0,runtime.iterations), ylims=(1e-10,0))
        plot!(iterations, residuals.Ux, yscale=:log10, label="Ux")
        plot!(iterations, residuals.Uy, yscale=:log10, label="Uy")
        plot!(iterations, residuals.p, yscale=:log10, label="p")
    end

    aero_eff = lift_to_drag(:foil, model, ρ, nu, α[1]) # Calculates lift-to-drag ratio

    # relocate XCALibre.jl vtk output files
    aero_eff_out = round(aero_eff,digits=3)
    α_out = round(α[1],digits=3)
    vtk_files = filter(x->endswith(x,".vtk"), readdir())
    for file ∈ vtk_files
        dest = "vtk_results/LD_Ratio = $(aero_eff_out), Alpha = $(α_out).vtk"
        mv(file, dest, force=true)
    end

    return aero_eff
end
```

Note that this code saves a single .vtk file of the last CFD iteration each time the optimiser samples the function. This .vtk file is then automatically renamed with that sample's results, and sorted into a /vtk_results subfolder (**which must be created before the optimisation is run**).

### Configure and run the Bayesian optimisation

Finally, the Bayesian optimiser must be configured before the optimisation can be performed. This example follows the BayesianOptimization.jl default configuration for the Gaussian process surrogate model, limiting input dimensions to 1 (the angle of attack). The surrogate model is then set to be optimised every 10 iterations. The inputs are limited to between 0 and 15 degrees. The problem is configured as a maximisation problem, with an initial sample period of 10 iterations and 50 maximum allowed iterations. The final line of the following code block is then run to perform the optimisation.

```julia
# Bayesian Optimisation (using BayesianOptimization.jl)
isdir("vtk_results") || mkdir("vtk_results")

# Initialises the Gaussian process surrogate model
model = ElasticGPE(1, #1 input dimension (α)
                   mean = MeanConst(0.0),
                   kernel = SEArd([0.0], 5.0),
                   capacity = 3000)
set_priors!(model.mean, [Normal(1, 2)])

modeloptimizer = MAPGPOptimizer(every = 10, maxeval = 40) # Optimises the Gaussian process every 10 iterations

# Optimisation Case Setup
opt = BOpt(foil_optim, # Function to be optimised - encloses the CFD case
            model, # Gaussian process surrogate model defined above
            UpperConfidenceBound(),
            modeloptimizer, # Model optimiser defined above
            [0.0], [15.0], # Minimum and maximum α constraints       
            repetitions = 1, # No repititions as CFD data is not noisy
            maxiterations = 50, # Maximum iterations
            sense = Max, # Maximisation problem
            initializer_iterations = 10, # No. of initial random samples
            verbosity = Progress)

opt = BOpt(foil_optim, # hide
            model, # hide
            UpperConfidenceBound(),
            modeloptimizer, # hide
            [0.0], [15.0], # hide      
            repetitions = 1, # hide
            maxiterations = 5, # hide
            sense = Max, # hide
            initializer_iterations = 2, # hide
            verbosity = Progress) # hide

result = boptimize!(opt) # Runs the optimisation procedure

using Pkg; Pkg.rm("BayesianOptimization", io=devnull) # hide

nothing # hide
"done"
```

# Example Optimisation Results
---

```@eval
using Plots

alpha_in = [2.8125, 10.3125, 14.0625, 6.5625, 4.6875, 12.1875, 8.4375, 0.9375, 1.40625, 8.90625, 6.779625965220729, 6.77968472431049, 6.7797314426495054, 6.77976068608606, 6.779786764662929, 6.7798083933298825, 6.779824452011973, 6.779839305833187, 6.779851985125758, 6.779864532650939, 6.7799660687298875, 6.7799701746255465, 6.779970295228141, 6.77997431759257, 6.779983415563036, 6.7799837395243685, 6.779983935151504, 6.779987648165338, 6.779990776177305, 6.779984858208577, 6.779969245384916, 6.779974853301299, 6.77996639334403, 6.779975137863877, 6.77996588653741, 6.7799787767924995, 6.779980463194295, 6.779982645115559, 6.779984123782711, 6.779985756585738, 6.780065815803057, 6.780066942543943, 6.780081725020423, 6.780063344501922, 6.780112623430458, 6.780065445290062, 6.780112028947505, 6.780088979322628, 6.78008982572309, 6.780090816324224, 6.852129317102274, 15.0, 6.755034985210623, 6.761466922699926, 6.766457867346942, 6.7701094609872365, 6.772562674953314, 6.774340290374087, 6.775562986686579, 6.776273742423618, 13.474012044415156, 12.361866906130413, 8.195985025525076, 5.334843671454779, 4.308592858166856, 2.6894603132241217, 2.858340694577276, 3.9867902638528654, 9.814658247047763, 1.932233467710785, 14.719640732627886, 9.682863385862596, 10.90134524899545, 12.877578634990728, 4.343344626917736]

aeroeff_out = [12.28662476568946, 15.996066573412422, 12.086660285887401, 18.095030718054726, 16.684759975625713, 14.101557983835567, 17.514651235093766, 4.6215245234556415, 6.787202918357864, 17.203294519181156, 18.1170435016882, 18.116757960309155, 18.117178450363166, 18.116980351431625, 18.116775418289507, 18.117332562156935, 18.116999501527403, 18.116946514089875, 18.11690912331511, 18.117181857003985, 18.117047847973517, 18.117081881041535, 18.117198176532845, 18.1172141350244, 18.117151423600156, 18.11701162747253, 18.117196229004325, 18.117302063623146, 18.117642349199684, 18.117122449593182, 18.11704990315254, 18.11745842192904, 18.1171359979369, 18.117344466780477, 18.117133217540022, 18.117057989337816, 18.116782216736464, 18.11721596351545, 18.117119958415003, 18.117170377304056, 18.117352502260307, 18.117163143088185, 18.117300338277143, 18.117066185177936, 18.11722389239519, 18.11713622798166, 18.116993107467152, 18.116991271960444, 18.117083277331048, 18.11714354615574, 18.097362115270933, 11.114298598972901, 18.108943497471795, 18.108780485320025, 18.108790519425263, 18.109037299059956, 18.108833040375593, 18.108767451351685, 18.1084445814309, 18.10857933786505, 12.690741887963036, 9.845212599830074, 17.636745115382112, 17.47574058124154, 16.058798546800677, 11.877430534208012, 12.4272778301407, 15.41709457750829, 16.462303743958294, 9.053215612096695, 11.382140258128832, 16.578166324395593, 15.519321514686421, 13.356080135399708, 16.116845772757824]

iteration = [1:25;]

scatter(
    alpha_in[1:25],aeroeff_out[1:25],xlabel="Angle of Attack, α [°]", ylabel="Aerodynamic Efficiency, Cl/Cd [-]",label="",
    legend=true,
    # marker=(:cross,4),
    zcolor=iteration,
    c=:linear_ternary_red_0_50_c52_n256,
    colorbar_title="Optimiser Iteration",
    xlims=[0,15.15], ylims=[0,20], clim=(0,25),
    frame_style=:box
    )

max_exp_alpha = ([6.479310344827586,6.479310344827586],[0,19])

plot!(max_exp_alpha,line=(:dash,:black),label="")

plot!(
    sort(alpha_in[1:50]), aeroeff_out[sortperm(alpha_in[1:50])],
    label="", line=(:solid,:black,0.5)
    )
annotate!(8.25, 2.5, Plots.text("Experimental\noptimum α", 10))
savefig("optimisation_iterations_vs_experiment.svg"); nothing # hide
nothing

fig1 = scatter(
    iteration[1:15],aeroeff_out[1:15],xlabel="Optimiser Iteration [-]", ylabel="Aerodynamic Efficiency, Cl/Cd [-]",label="",
    ylim = (0,20),
    legend=true,# marker=(:cross,4,:black)
    ) #L/D against iteration number graph

fig2 = scatter(iteration[1:15],alpha_in[1:15],xlabel="Optimiser Iteration [-]", ylabel="Angle of Attack, α [°]",label="",
ylim = (0,15),
legend=true, #marker=(:cross,4,:black)
) #α against iteration number graph


# exp_alpha = [0.1724137931034484, 2.068965517241379, 4.482758620689655, 6.479310344827586, 8.441064638783269, 10.344827586206897, 11.206896551724139, 12.551020408163264, 14.13793103448276]
# exp_aeroeff = [8.872377622377625, 36.656207598371786, 52.94361173623201, 53.41160498793243, 53.10395622895623, 50.95347249317206, 48.78298051001016, 4.906565656565657, 3.354085331846068]
# fig2 = scatter(exp_alpha,exp_aeroeff,xlabel="Angle of Attack, α [°]", ylabel="Aerodynamic Efficiency, Cl/Cd [-]",title="Experimental Data (NACA0012, Re=500,000)",
# legend=false,marker=(:circle,4,:black)) #Experimental data

plot(fig1, fig2, frame_style=:box)

savefig("optimisation_vs_iteration.svg"); nothing # hide
nothing

# output

```
![](optimisation_iterations_vs_experiment.svg)

![](optimisation_vs_iteration.svg)
