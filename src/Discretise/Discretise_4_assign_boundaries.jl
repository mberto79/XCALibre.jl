export assign, @assign!



# Assign function definition for vector field
assign(vec::VectorField, args...) = begin
    # Retrieve user selected float type and boundaries
    mesh = vec.mesh
    float = _get_float(mesh)
    mesh_temp = adapt(CPU(), mesh) # WARNING: Temp solution 
    boundaries = mesh_temp.boundaries
    boundary_info = boundary_map(mesh)

    # Assign tuples for boundary condition vectors
    @reset vec.x.BCs = ()
    @reset vec.y.BCs = ()
    @reset vec.z.BCs = ()
    @reset vec.BCs = ()

    # Loop over boundary condition arguments to set boundary condition vectors
    for arg ∈ args

        # Set boundary index and retrieve corresponding name
        idx = boundary_index(boundary_info, arg.ID)
        bname = boundaries[idx].name
        println("Setting boundary $idx: ", bname)

        # Exception 1: value is vector
        if typeof(arg.value) <: AbstractVector
            # Error check if vector is 3 elements
            length(arg.value) == 3 || throw("Vector must have 3 components")
            # Set boundary conditions
            xBCs = (vec.x.BCs..., fixedValue(arg, idx, float(arg.value[1])))
            yBCs = (vec.y.BCs..., fixedValue(arg, idx, float(arg.value[2])))
            zBCs = (vec.z.BCs..., fixedValue(arg, idx, float(arg.value[3])))
            uBCs = (vec.BCs..., fixedValue(arg, idx, float.(arg.value)))
            @reset vec.x.BCs = xBCs
            @reset vec.y.BCs = yBCs
            @reset vec.z.BCs = zBCs
            @reset vec.BCs = uBCs
        elseif typeof(arg.value) <: NamedTuple
            # Set boundary conditions
            xBCs = (vec.x.BCs..., fixedValue(arg, idx, arg.value))
            yBCs = (vec.y.BCs..., fixedValue(arg, idx, arg.value))
            zBCs = (vec.z.BCs..., fixedValue(arg, idx, arg.value))
            uBCs = (vec.BCs..., fixedValue(arg, idx, arg.value))
            @reset vec.x.BCs = xBCs
            @reset vec.y.BCs = yBCs
            @reset vec.z.BCs = zBCs
            @reset vec.BCs = uBCs
        elseif typeof(arg.value) <: Function
            # Set boundary conditions
            xBCs = (vec.x.BCs..., fixedValue(arg, idx, arg.value))
            yBCs = (vec.y.BCs..., fixedValue(arg, idx, arg.value))
            zBCs = (vec.z.BCs..., fixedValue(arg, idx, arg.value))
            uBCs = (vec.BCs..., fixedValue(arg, idx, arg.value))
            @reset vec.x.BCs = xBCs
            @reset vec.y.BCs = yBCs
            @reset vec.z.BCs = zBCs
            @reset vec.BCs = uBCs
        elseif typeof(arg.value) <: XCALibreUserFunctor
            # Set boundary conditions
            xBCs = (vec.x.BCs..., fixedValue(arg, idx, arg.value))
            yBCs = (vec.y.BCs..., fixedValue(arg, idx, arg.value))
            zBCs = (vec.z.BCs..., fixedValue(arg, idx, arg.value))
            uBCs = (vec.BCs..., fixedValue(arg, idx, arg.value))
            @reset vec.x.BCs = xBCs
            @reset vec.y.BCs = yBCs
            @reset vec.z.BCs = zBCs
            @reset vec.BCs = uBCs
        else
            # Set boundary conditions
            xBCs = (vec.x.BCs..., fixedValue(arg, idx, float(arg.value)))
            yBCs = (vec.y.BCs..., fixedValue(arg, idx, float(arg.value)))
            zBCs = (vec.z.BCs..., fixedValue(arg, idx, float(arg.value)))
            uBCs = (vec.BCs..., fixedValue(arg, idx, float(arg.value)))
            @reset vec.x.BCs = xBCs
            @reset vec.y.BCs = yBCs
            @reset vec.z.BCs = zBCs
            @reset vec.BCs = uBCs
        end
    end
    return vec
end

# Assign function definition for scalar field
assign(scalar::ScalarField, args...) = begin

    # Retrieve user selected float type and boundaries
    mesh = scalar.mesh
    float = _get_float(mesh)
    mesh_temp = adapt(CPU(), mesh) # WARNING: Temp solution 
    boundaries = mesh_temp.boundaries
    boundary_info = boundary_map(mesh)

    # Assign tuples for boundary condition scalar
    @reset scalar.BCs = ()

    # Loop over boundary condition arguments to set boundary condition scalar
    for arg ∈ args

        # Set boundary index and retrieve corresponding name
        idx = boundary_index(boundary_info, arg.ID)
        bname = boundaries[idx].name
        println("Setting boundary $idx: ", bname)

        # Exception 1: value is a number
        if typeof(arg.value) <: Number
            BCs = (fixedValue(arg, idx, float(arg.value))) # doesn't work with tuples
            @reset scalar.BCs = (scalar.BCs..., BCs)

        # Exception 2: value is a named tuple (used in wall functions)
        elseif typeof(arg.value) <: NamedTuple
            BCs_vals = arg.value
            BCs = (fixedValue(arg, idx, BCs_vals))
            @reset scalar.BCs = (scalar.BCs..., BCs)
            
        # Exception 3: value is a function
        elseif typeof(arg.value) <: Function
            BCs_vals = arg.value
            BCs = (fixedValue(arg, idx, BCs_vals))
            @reset scalar.BCs = (scalar.BCs..., BCs)

        # Error exception: Value is not named tuple or number
        else
            error("Value given to boundary $idx ($bname) is not recognised")
        end
    end
    return scalar
end

# Laminar assign macro definition
macro assign!(model, field, BCs)
    # Retrieve defined model, field and boundary conditions
    emodel = esc(model)
    efield = Symbol(field)
    eBCs = esc(BCs)
    
    # Assign boundary conditions to model
    quote
        f = $emodel.$efield
        f = assign(f, $eBCs...)
        $emodel = @set $emodel.$efield = f
    end
end

# Turbulent assign macro definition
macro assign!(model, turb, field, BCs)
    # Retrieve defined model, field and boundary conditions
    emodel = esc(model)
    eturb = Symbol(turb)
    efield = Symbol(field)
    eBCs = esc(BCs)

    # Assign boundary conditions to model
    quote
        f = $emodel.$eturb.$efield
        f = assign(f, $eBCs...)
        $emodel = @set $emodel.$eturb.$efield = f
    end
end

# Set schemes function definition with default set variables
"""
    set_schemes(;
        # keyword arguments and their default values
        time=SteadyState,
        divergence=Linear, 
        laplacian=Linear, 
        gradient=Orthogonal) = begin
        
        # Returns NamedTuple definition for scheme 
        (
            time=time,
            divergence=divergence,
            laplacian=laplacian,
            gradient=gradient
        )
    end

The `set_schemes` function is used at the top-level API to help users define discretisation schemes for every field solved. It offers default values, thus users can pick and choose which entry they wish to modify.

# inputs

- `time` is used to set the time schemes (default is `SteadyState`)
- `divergence` is used to set the divergence scheme (default is `Linear`) 
- `laplacian` is used to set the laplacian scheme (default is `Linear`)
- `gradient`  is used to set the gradient scheme (default is `Orthogonal`)

"""
set_schemes(;
    # keyword arguments and their default values
    time=SteadyState,
    divergence=Linear, 
    laplacian=Linear, 
    gradient=Orthogonal) = begin
    
    # Returns NamedTuple definition for scheme 
    (
        time=time,
        divergence=divergence,
        laplacian=laplacian,
        gradient=gradient
    )
end