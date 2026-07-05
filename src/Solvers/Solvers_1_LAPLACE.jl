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
    output=VTK(), pref=nothing, ncorrectors=0, inner_loops=0,
    petsc_options="", solve_on=nothing
    )

    residuals = setup_laplace_solver(
        LAPLACE, model, config;
        output=output,
        pref=pref,
        ncorrectors=ncorrectors,
        inner_loops=inner_loops,
        petsc_options=petsc_options, solve_on=solve_on
        )

    return residuals
end


function setup_laplace_solver(
    solver_variant, model, config;
    output=VTK(), pref=nothing, ncorrectors=0, inner_loops=0,
    petsc_options="", solve_on=nothing
    )

    (; solvers, schemes, runtime, hardware, boundaries) = config
    (; iterations, write_interval, dt) = runtime
    (; backend) = hardware

    (; T, Tf) = model.energy

    (; k, kf, cp, rho, rhocp, rDf) = model.solid

    mesh = model.domain
    assert_distributable(mesh, boundaries) # rejects periodic BCs on a DistributedMesh


    source_field = ScalarField(mesh) #0.0 field


    @info "Defining models..."
    T_eqn = (
        Time{schemes.time}(rhocp, T) #0.0 by default
        - Laplacian{schemes.laplacian}(rDf, T)
        ==
        - Source(source_field)
    ) → ScalarEquation(T, boundaries.T)

    # Krylov preconditioner/workspace are serial-only (distributed uses PETSc PCs)
    if !is_distributed_mesh(mesh)
        @info "Initialising preconditioners..."
        @reset T_eqn.preconditioner = set_preconditioner(solvers.preconditioner, T_eqn)

        @info "Pre-allocating solvers..."
        @reset T_eqn.solver = _workspace(solvers.solver, _b(T_eqn))
    end

    @info "Initialising energy model..."
    energyModel = initialise(model.energy, model, T, rDf, rhocp, k, kf, cp, rho, config)

    # wrap for the linear-solve seam (identity serial / DistributedEqn on a DistributedMesh)
    T_eqn = wrap_eqn(T_eqn, mesh, solvers, config; petsc_options, solve_on)


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

    (; T, Tf) = model.energy
    (; k, kf, cp, rho, rhocp, rDf) = model.solid
    mesh = model.domain
    (; solvers, schemes, runtime, hardware, boundaries, postprocess) = config
    (; iterations, write_interval, dt) = runtime
    (; backend) = hardware

    distributed = is_distributed_mesh(mesh)

    dt_cpu = zeros(_get_float(mesh), 1)
    copyto!(dt_cpu, config.runtime.dt)

    postprocess = convert_time_to_iterations(postprocess,model,dt_cpu[1],iterations)
    @info "Starting LAPLACE loops..."
    progress = distributed ? nothing : Progress(iterations; dt=1.0, showspeed=true)

    sync!(T, mesh, config) # prime ghosts (no-op serial)
    for iteration ∈ 1:iterations
        time = iteration *dt

        rt = solve_equation!(T_eqn, T, boundaries.T, solvers, config; time=time)

        if typeof(model.solid) <: NonUniform
            energy!(model.energy, model, T, rDf, rhocp, k, kf, cp, rho, config)
        end

        R_T[iteration] = rt

        if (R_T[iteration] <= solvers.convergence) && (typeof(model.time) <: Steady)
            if !distributed
                progress.n = iteration
                finish!(progress)
            end
            is_report_rank(mesh) && @info "Simulation converged in $iteration iterations!"
            if !signbit(write_interval)
                outputWriter === nothing || save_output(model, outputWriter, iteration, time, config)
            end

            break
        end

        distributed || ProgressMeter.next!(
            progress, showvalues = [
                (:time, iteration*dt_cpu[1]),
                (:T_residual, R_T[iteration])
                ]
            )

        distributed || runtime_postprocessing!(postprocess,iteration,iterations,nothing,time,config)
        if iteration%write_interval + signbit(write_interval) == 0
            outputWriter === nothing || save_output(model, outputWriter, iteration, time, config)
            distributed || save_postprocessing(postprocess,iteration,time,mesh,outputWriter,config.boundaries)
        end

    end # end for loop

    return (T=R_T,)
end

function ModelPhysics.save_output(model::Physics{T,F,SO,M,Tu,E,D,BI}, outputWriter, iteration, time, config
    ) where {T,F,SO,M,Tu,E<:Conduction,D,BI}
    
    args = (
        ("T", model.energy.T),
    )
    write_results(iteration, time, model.domain, outputWriter, config.boundaries, args...)
end