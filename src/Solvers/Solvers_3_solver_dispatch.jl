export run!

"""
    residuals = run!(model::Physics, config::Configuration)

Description goes here

# Input arguments

# Example

```julia
Something here
```

"""
run!() = nothing # dummy function for providing general documentation

# Incompressible solver (steady)
"""
    run!(
    model::Physics{T,F,M,Tu,E,D,BI}, config; pref=nothing
    ) where{T<:Steady,F<:Incompressible,M,Tu,E,D,BI}

Incompressible steady solver using the SIMPLE algorithm.

# Input
- `model`  -- Physics model defiend by user and passed to run!.
- `config` -- Configuration structure defined by user with solvers, schemes, runtime and 
                hardware structures set.
- `pref`   -- Reference pressure value for cases that do not have a pressure defining BC.

# Output
- `R_ux`  - Vector of x-velocity residuals for each iteration.
- `R_uy`  - Vector of y-velocity residuals for each iteration.
- `R_uz`  - Vector of y-velocity residuals for each iteration.
- `R_p`   - Vector of pressure residuals for each iteration.
- `model` - Physics model output including field parameters.
"""
run!(
    model::Physics{T,F,M,Tu,E,D,BI}, config; pref=nothing
    ) where{T<:Steady,F<:Incompressible,M,Tu,E,D,BI} = 
begin
    residuals = simple!(model, config, pref=pref)
    return residuals
end

# Incompressible solver (transient)
"""
    run!(
    model::Physics{T,F,M,Tu,E,D,BI}, config; pref=nothing
    ) where{T<:Transient,F<:Incompressible,M,Tu,E,D,BI}

Incompressible unsteady solver using the PISO algorithm.

# Input
- `model`  -- Physics model defined by user and passed to run!.
- `config` -- Configuration structure defined by user with solvers, schemes, runtime and 
                hardware structures set.
- `pref`   -- Reference pressure value for cases that do not have a pressure defining BC.

# Output
- `R_ux`  - Vector of x-velocity residuals for each iteration.
- `R_uy`  - Vector of y-velocity residuals for each iteration.
- `R_uz`  - Vector of y-velocity residuals for each iteration.
- `R_p`   - Vector of pressure residuals for each iteration.
- `model` - Physics model output including field parameters.
"""
run!(
    model::Physics{T,F,M,Tu,E,D,BI}, config; pref=nothing
    ) where{T<:Transient,F<:Incompressible,M,Tu,E,D,BI} = 
begin
    residuals = piso!(model, config, pref=pref); #, pref=0.0)
    return residuals
end

# Weakly Compressible solver (steady)
"""
    run!(
    model::Physics{T,F,M,Tu,E,D,BI}, config; pref=nothing
    ) where{T<:Steady,F<:WeaklyCompressible,M,Tu,E,D,BI}

Mildly compressible steady solver using the SIMPLE algorithm for low speed cases with heat 
    transfer.

# Input
- `model`  -- Physics model defiend by user and passed to run!.
- `config` -- Configuration structure defined by user with solvers, schemes, runtime and 
                hardware structures set.
- `pref`   -- Reference pressure value for cases that do not have a pressure defining BC.

# Output
- `R_ux`  - Vector of x-velocity residuals for each iteration.
- `R_uy`  - Vector of y-velocity residuals for each iteration.
- `R_uz`  - Vector of y-velocity residuals for each iteration.
- `R_p`   - Vector of pressure residuals for each iteration.
- `R_e`   - Vector of energy residuals for each iteration.
- `model` - Physics model output including field parameters.
"""
run!(
    model::Physics{T,F,M,Tu,E,D,BI}, config; pref=nothing
    ) where{T<:Steady,F<:WeaklyCompressible,M,Tu,E,D,BI} = 
begin
    residuals = csimple!(model, config, pref=pref); #, pref=0.0)
    return residuals
end

# Compressible solver (steady)
run!(
    model::Physics{T,F,M,Tu,E,D,BI}, config; pref=nothing
    ) where{T<:Steady,F<:Compressible,M,Tu,E,D,BI} = 
begin
    residuals = csimple!(model, config, pref=pref); #, pref=0.0)
    return residuals
end

# Weakly Compressible solver (transient)
"""
    run!(
    model::Physics{T,F,M,Tu,E,D,BI}, config; pref=nothing
    ) where{T<:Transient,F<:WeaklyCompressible,M,Tu,E,D,BI}

Mildly compressible unsteady solver using the PISO algorithm for low speed cases with heat 
    transfer.

### Input
- `model`  -- Physics model defiend by user and passed to run!.
- `config` -- Configuration structure defined by user with solvers, schemes, runtime and 
                hardware structures set.
- `pref`   -- Reference pressure value for cases that do not have a pressure defining BC.

### Output
- `R_ux`  - Vector of x-velocity residuals for each iteration.
- `R_uy`  - Vector of y-velocity residuals for each iteration.
- `R_uz`  - Vector of y-velocity residuals for each iteration.
- `R_p`   - Vector of pressure residuals for each iteration.
- `R_e`   - Vector of energy residuals for each iteration.
- `model` - Physics model output including field parameters.
"""
run!(
    model::Physics{T,F,M,Tu,E,D,BI}, config
    ) where{T<:Transient,F<:WeaklyCompressible,M,Tu,E,D,BI} = 
begin
    residuals = piso_comp!(model, config); #, pref=0.0)
    return residuals
end

# Compressible solver (transient)
run!(
    model::Physics{T,F,M,Tu,E,D,BI}, config
    ) where{T<:Transient,F<:Compressible,M,Tu,E,D,BI} = 
begin
    residuals = piso_comp!(model, config); #, pref=0.0)
    return residuals
end