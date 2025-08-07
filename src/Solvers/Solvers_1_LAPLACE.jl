export laplace!
export LAPLACE
export setup_laplace_solver



"""
    laplace!(model_in, config; 
        output=VTK(), pref=nothing, ncorrectors=0, inner_loops=0)


Top-level entry point for solving the Laplace (heat conduction) equation on `model.domain`.  
Optionally runs in steady or transient mode and can call Conduction energy model for high fidelity (k and cp are recomputed at each iteration).


# Input arguments

- `model` reference to a `Physics` model defined by the user.
- `config` Configuration structure defined by the user with solvers, schemes, runtime and hardware structures configuration details.
- `output` select the format used for simulation results from `VTK()` or `OpenFOAM` (default = `VTK()`)
- `pref` Reference pressure value for cases that do not have a pressure defining BC. Incompressible solvers only (default = `nothing`)
- `ncorrectors` number of non-orthogonality correction loops (default = `0`)
- `inner_loops` number to inner loops used in transient solver based on PISO algorithm (default = `0`)

# Output

This function returns a `NamedTuple` for accessing the residuals (e.g. `residuals.Ux`) with the following entries:

- `T` Vector of T residuals for each iteration.

"""
function laplace!(
    model, config; 
    output=VTK(), pref=nothing, ncorrectors=0, inner_loops=0
    )

    residuals = setup_laplace_solver(
        LAPLACE, model, config;
        output=output,
        pref=pref, 
        ncorrectors=ncorrectors, 
        inner_loops=inner_loops
        )

    return residuals
end


function setup_laplace_solver(
    solver_variant, model, config; 
    output=VTK(), pref=nothing, ncorrectors=0, inner_loops=0
    ) 

    (; solvers, schemes, runtime, hardware, boundaries) = config
    (; iterations, write_interval, dt) = runtime
    (; backend) = hardware

    (; T, Tf, rDf, rhocp, k, kf, cp, rho) = model.energy

    mesh = model.domain

    source_field = ScalarField(mesh) #0.0 field
   

    @info "Defining models..."

    if typeof(model.time) <: Transient
        @info "Transient"
        T_eqn = (
            Time{schemes.time}(rhocp, T)
            - Laplacian{schemes.laplacian}(rDf, T)
            ==
            - Source(source_field)
        ) → ScalarEquation(T, boundaries.T)
    elseif typeof(model.time) <: Steady
        @info "Steady"
        T_eqn = (
            - Laplacian{schemes.laplacian}(rDf, T)
            ==
            - Source(source_field)
        ) → ScalarEquation(T, boundaries.T)
    end

    @info "Initialising preconditioners..."

    @reset T_eqn.preconditioner = set_preconditioner(solvers.preconditioner, T_eqn)

    @info "Pre-allocating solvers..."

    @reset T_eqn.solver = _workspace(solvers.solver, _b(T_eqn))

    if typeof(model.energy) <: Conduction
        @info "Initialising energy model..."
        energyModel = initialise(model.energy, model, T, rDf, rhocp, k, kf, cp, rho, model.energy.material, config)
    end


    # The part that was previously inside the solver
    
    outputWriter = initialise_writer(output, model.domain) 
    interpolate!(Tf, T, config)

    @info "Allocating working memory..."

    n_cells = length(mesh.cells)
    TF = _get_float(mesh)
    prev = KernelAbstractions.zeros(backend, TF, n_cells) 
    R_T = ones(TF, iterations)
    
    # Initial calculations
    time = zero(TF) # assuming time=0


    residuals  = solver_variant(
        model, T_eqn, config; 
        output=output,
        pref=pref, 
        ncorrectors=ncorrectors, 
        inner_loops=inner_loops,
        outputWriter, R_T, time)

    return residuals
end

function LAPLACE(
    model, T_eqn, config;
    output=VTK(), pref=nothing, ncorrectors=0, inner_loops=0,
    outputWriter, R_T, time
    )

    if typeof(model.energy) <: Conduction
        (; T, Tf, rDf, rhocp, k, kf, cp, rho, material) = model.energy
    else
        (; T, Tf) = model.energy
    end

    (; solvers, schemes, runtime, hardware, boundaries) = config
    (; iterations, write_interval, dt) = runtime
    (; backend) = hardware



    @info "Starting LAPLACE loops..."
    progress = Progress(iterations; dt=1.0, showspeed=true)

    @time for iteration ∈ 1:iterations
        time = iteration *dt

        rt = solve_equation!(T_eqn, T, boundaries.T, solvers, config; time=time)
        
        if typeof(model.energy) <: Conduction
            energy!(model.energy, model, T, rDf, rhocp, k, kf, cp, rho, material, config)
        end

        R_T[iteration] = rt

        if (R_T[iteration] <= solvers.convergence) && (typeof(model.time) <: Steady)
            progress.n = iteration
            finish!(progress)
            @info "Simulation converged in $iteration iterations!"
            if !signbit(write_interval) 
                save_output(model, outputWriter, iteration, time, config)
            end
            
            break
        end

        ProgressMeter.next!(
            progress, showvalues = [
                (:time, iteration*runtime.dt),
                (:T_residual, R_T[iteration])
                ]
            )

        if iteration%write_interval + signbit(write_interval) == 0      
            save_output(model, outputWriter, iteration, time, config)
        end

    end # end for loop
    
    return (T=R_T)
end