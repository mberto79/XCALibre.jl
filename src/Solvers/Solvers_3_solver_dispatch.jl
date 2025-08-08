export run!

"""
    function run!(
        model::Physics, config; 
        output=VTK(), pref=nothing, ncorrectors=0, inner_loops=0
        )

        # here an internal function is used for solver dispatch
        return residuals
    end

This is the top level API function to initiate a simulation. It uses the user-provided `model` defined as a `Physics` object to dispatch to the appropriate solver.

# Dispatched flow solvers

- Steady incompressible (SIMPLE algorithm for coupling)
- Transient incompressible (PISO algorithm for coupling)
- Steady weakly compressible (SIMPLE algorithm for coupling)
- Transient weakly compressible (PISO algorithm for coupling)

# Input arguments
- `model` reference to a `Physics` model defined by the user.
- `config` Configuration structure defined by the user with solvers, schemes, runtime and hardware structures configuration details.
- `output` select the format used for simulation results from `VTK()` or `OpenFOAM()` (default = `VTK()`)
- `pref` Reference pressure value for cases that do not have a pressure defining BC. Incompressible solvers only (default = `nothing`)
- `ncorrectors` number of non-orthogonality correction loops (default = `0`)
- `inner_loops` number to inner loops used in transient solver based on PISO algorithm (default = `0`)

# Output

This function returns a `NamedTuple` for accessing the residuals (e.g. `residuals.Ux`). The fields available within the returned `residuals` tuple depend on the solver used. For example, for an incompressible solver, a x-momentum equation residual can be retrieved accessing the `Ux` field i.e. `residuals.Ux`. Look at reference guide for each dispatch method to find out which fields are available.

# Example 

```julia
residuals = run!(model, config) 

# to access the pressure residual

residuals.p 
```

"""
run!() = nothing # dummy function for providing general documentation

# Incompressible solver (steady)
"""
    run!(
        model::Physics{T,F,M,Tu,E,D,BI}, config;
        output=VTK(), pref=nothing, ncorrectors=0, inner_loops=0
        ) where{T<:Steady,F<:Incompressible,M,Tu,E,D,BI} = 
    begin
        residuals = simple!(model, config, pref=pref)
        return residuals
    end

Calls the incompressible steady solver using the SIMPLE algorithm.

# Input
- `model` represents the `Physics` model defined by user.
- `config` Configuration structure defined by user with solvers, schemes, runtime and hardware structures configuration details.
- `output` select the format used for simulation results from `VTK()` or `OpenFOAM()` (default = `VTK()`)
- `pref` Reference pressure value for cases that do not have a pressure defining BC. Incompressible solvers only.

# Output

This function returns a `NamedTuple` for accessing the residuals (e.g. `residuals.Ux`) with the following entries:

- `Ux`  - Vector of x-velocity residuals for each iteration.
- `Uy`  - Vector of y-velocity residuals for each iteration.
- `Uz`  - Vector of y-velocity residuals for each iteration.
- `p`   - Vector of pressure residuals for each iteration.
"""
run!(
    model::Physics{T,F,M,Tu,E,D,BI}, config; 
    output=VTK(), pref=nothing, ncorrectors=0, inner_loops=0
    ) where{T<:Steady,F<:Incompressible,M,Tu,E,D,BI} = 
begin
    residuals = simple!(
        model, config, 
        output=output,
        pref=pref, 
        ncorrectors=ncorrectors, 
        inner_loops=inner_loops
        )
    return residuals
end

# Incompressible solver (transient)
"""
    run!(
        model::Physics{T,F,M,Tu,E,D,BI}, config; 
        output=VTK(), pref=nothing, ncorrectors=0, inner_loops=0
        ) where{T<:Transient,F<:Incompressible,M,Tu,E,D,BI} = 
    begin
        residuals = piso!(model, config, pref=pref); #, pref=0.0)
        return residuals
    end

Calls the incompressible transient solver using the PISO algorithm.

# Input

- `model` represents the `Physics` model defined by user.
- `config` Configuration structure defined by user with solvers, schemes, runtime and hardware structures configuration details.
- `output` select the format used for simulation results from `VTK()` or `OpenFOAM()` (default = `VTK()`)
- `pref` Reference pressure value for cases that do not have a pressure defining BC. Incompressible solvers only.

# Output

This function returns a `NamedTuple` for accessing the residuals (e.g. `residuals.Ux`) with the following entries:

- `Ux`  - Vector of x-velocity residuals for each iteration.
- `Uy`  - Vector of y-velocity residuals for each iteration.
- `Uz`  - Vector of y-velocity residuals for each iteration.
- `p`   - Vector of pressure residuals for each iteration.

"""
run!(
    model::Physics{T,F,M,Tu,E,D,BI}, config; 
    output=VTK(), pref=nothing, ncorrectors=0, inner_loops=2
    ) where{T<:Transient,F<:Incompressible,M,Tu,E,D,BI} = 
begin
    residuals = piso!(
        model, config, 
        output=output,
        pref=pref, 
        ncorrectors=ncorrectors, 
        inner_loops=inner_loops
        )
    return residuals
end

# Weakly Compressible solver (steady)
"""
    run!(
        model::Physics{T,F,M,Tu,E,D,BI}, config; 
        output=VTK(), pref=nothing, ncorrectors=0, inner_loops=0
        ) where{T<:Steady,F<:WeaklyCompressible,M,Tu,E,D,BI} = 
    begin
        residuals = csimple!(model, config, pref=pref); #, pref=0.0)
        return residuals
    end

Calls the compressible steady solver using the SIMPLE algorithm for weakly compressible fluids.

# Input

- `model` represents the `Physics` model defined by user.
- `config` Configuration structure defined by user with solvers, schemes, runtime and hardware structures configuration details.
- `output` select the format used for simulation results from `VTK()` or `OpenFOAM()` (default = `VTK()`)
- `pref` Reference pressure value for cases that do not have a pressure defining BC. Incompressible solvers only.

# Output

This function returns a `NamedTuple` for accessing the residuals (e.g. `residuals.Ux`) with the following entries:

- `Ux`  - Vector of x-velocity residuals for each iteration.
- `Uy`  - Vector of y-velocity residuals for each iteration.
- `Uz`  - Vector of y-velocity residuals for each iteration.
- `p`   - Vector of pressure residuals for each iteration.
- `e`   - Vector of energy residuals for each iteration.

"""
run!(
    model::Physics{T,F,M,Tu,E,D,BI}, config; 
    output=VTK(), pref=nothing, ncorrectors=0, inner_loops=0
    ) where{T<:Steady,F<:WeaklyCompressible,M,Tu,E,D,BI} = 
begin
    residuals = csimple!(
        model, config, 
        output=output,
        pref=pref, 
        ncorrectors=ncorrectors, 
        inner_loops=inner_loops
        )
    return residuals
end

# Compressible solver (steady)
run!(
    model::Physics{T,F,M,Tu,E,D,BI}, config; 
    output=VTK(), pref=nothing, ncorrectors=0, inner_loops=0
    ) where{T<:Steady,F<:Compressible,M,Tu,E,D,BI} = 
begin
    residuals = csimple!(
        model, config, 
        output=output,
        pref=pref, 
        ncorrectors=ncorrectors, 
        inner_loops=inner_loops
        )
    return residuals
end

# Weakly Compressible solver (transient)
"""
    run!(
        model::Physics{T,F,M,Tu,E,D,BI}; 
        output=VTK(), pref=nothing, ncorrectors=0, inner_loops=0
        ) where{T<:Transient,F<:WeaklyCompressible,M,Tu,E,D,BI} = 
    begin
        residuals = cpiso!(model, config)
        return residuals
    end

Calls the compressible transient solver using the PISO algorithm for weakly compressible fluids.

# Input
- `model` represents the `Physics` model defined by user.
- `config` Configuration structure defined by user with solvers, schemes, runtime and hardware structures configuration details.
- `output` select the format used for simulation results from `VTK()` or `OpenFOAM()` (default = `VTK()`)
- `pref` Reference pressure value for cases that do not have a pressure defining BC. Incompressible solvers only.

# Output

This function returns a `NamedTuple` for accessing the residuals (e.g. `residuals.Ux`) with the following entries:

- `Ux`  - Vector of x-velocity residuals for each iteration.
- `Uy`  - Vector of y-velocity residuals for each iteration.
- `Uz`  - Vector of y-velocity residuals for each iteration.
- `p`   - Vector of pressure residuals for each iteration.
- `e`   - Vector of energy residuals for each iteration.
"""
run!(
    model::Physics{T,F,M,Tu,E,D,BI}, config;
    output=VTK(), pref=nothing, ncorrectors=0, inner_loops=2
    ) where{T<:Transient,F<:WeaklyCompressible,M,Tu,E,D,BI} = 
begin
    residuals = cpiso!(
        model, config, 
        output=output,
        pref=pref, 
        ncorrectors=ncorrectors, 
        inner_loops=inner_loops
        )
    return residuals
end

# Compressible solver (transient)
run!(
    model::Physics{T,F,M,Tu,E,D,BI}, config;
    output=VTK(), pref=nothing, ncorrectors=0, inner_loops=2
    ) where{T<:Transient,F<:Compressible,M,Tu,E,D,BI} = 
begin
    residuals = cpiso!(
        model, config, 
        output=output,
        pref=pref, 
        ncorrectors=ncorrectors, 
        inner_loops=inner_loops
        )
    return residuals
end