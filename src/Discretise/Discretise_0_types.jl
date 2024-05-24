export AbstractScheme, AbstractBoundary
export AbstractDirichlet, AbstractNeumann
export Dirichlet, fixedValue, Neumann
export KWallFunction, OmegaWallFunction, NutWallFunction
export Constant, Linear, Upwind
export Steady, Euler, CrankNicolson
export Orthogonal, Midpoint
export assign, @assign!
export set_schemes

# SUPPORTED DISCRETISATION SCHEMES 

abstract type AbstractScheme end
struct Constant <: AbstractScheme end
struct Linear <: AbstractScheme end
struct Upwind <: AbstractScheme end
struct Orthogonal <: AbstractScheme end
struct Midpoint <: AbstractScheme end
struct Steady <: AbstractScheme end 
struct Euler <: AbstractScheme end 
struct CrankNicolson <: AbstractScheme end



# SUPPORTED BOUNDARY CONDITIONS 

abstract type AbstractBoundary end
abstract type AbstractDirichlet <: AbstractBoundary end
abstract type AbstractNeumann <: AbstractBoundary end
abstract type AbstractWallFunction <: AbstractDirichlet end

# Dirichlet structure and constructor function
struct Dirichlet{I,V} <: AbstractDirichlet
    ID::I
    value::V
end
Adapt.@adapt_structure Dirichlet
function fixedValue(BC::Dirichlet, ID::I, value::V) where {I<:Integer,V}
    # Exception 1: Value is scalar
    if V <: Number
        return Dirichlet{I,eltype(value)}(ID, value)
    # Exception 2: value is vector
    elseif V <: Vector
        if length(value) == 3 
            nvalue = SVector{3, eltype(value)}(value)
            return Dirichlet{I,typeof(nvalue)}(ID, nvalue)
        # Error statement if vector is invalid
        else
            throw("Only vectors with three components can be used")
        end
    # Error if value is not scalar or vector
    else
        throw("The value provided should be a scalar or a vector")
    end
end

# Neumann structure and constructor function
struct Neumann{I,V} <: AbstractBoundary
    ID::I 
    value::V 
end
Adapt.@adapt_structure Neumann
function fixedValue(BC::Neumann, ID::I, value::V) where {I<:Integer,V}
    # Exception 1: value is scalar
    if V <: Number
        return Neumann{I,eltype(value)}(ID, value)
    # Exception 2: value is vector
    elseif V <: Vector
        if length(value) == 3 
            nvalue = SVector{3, eltype(value)}(value)
            return Neumann{I,typeof(nvalue)}(ID, nvalue)
        # Error statement if vector is invalid        
        else
            throw("Only vectors with three components can be used")
        end
    # Error if value is not scalar or vector
    else
        throw("The value provided should be a scalar or a vector")
    end
end

# Kwall function structure and constructor
struct KWallFunction{I,V} <: AbstractWallFunction
    ID::I 
    value::V 
end
Adapt.@adapt_structure KWallFunction
KWallFunction(name::Symbol) = begin
    KWallFunction(name, (kappa=0.41, beta1=0.075, cmu=0.09, B=5.2, E=9.8))
end

# Omega wall function structure and constructor
struct OmegaWallFunction{I,V} <: AbstractWallFunction
    ID::I 
    value::V 
end
Adapt.@adapt_structure OmegaWallFunction
OmegaWallFunction(name::Symbol) = begin
    OmegaWallFunction(name, (kappa=0.41, beta1=0.075, cmu=0.09, B=5.2, E=9.8))
end
function fixedValue(BC::OmegaWallFunction, ID::I, value::V) where {I<:Integer,V}
    # Exception 1: Value is scalar
    if V <: Number
        return OmegaWallFunction{I,typeof(value)}(ID, value)
        # Exception 2: value is a tupple
    elseif V <: NamedTuple
        return OmegaWallFunction{I,V}(ID, value)
    # Error if value is not scalar or tuple
    else
        throw("The value provided should be a scalar or a tuple")
    end
end

# Nut wall function structure and constructor
struct NutWallFunction{I,V} <: AbstractWallFunction 
    ID::I 
    value::V 
end
Adapt.@adapt_structure NutWallFunction
NutWallFunction(name::Symbol) = begin
    NutWallFunction(name, (kappa=0.41, beta1=0.075, cmu=0.09, B=5.2, E=9.8))
end

# Assign function definition for vector field
assign(vec::VectorField, model, args...) = begin
    # Retrieve user selected float type and boundaries
    float = _get_float(vec.mesh)
    boundaries = vec.mesh.boundaries

    # Assign tuples for boundary condition vectors
    @reset vec.x.BCs = ()
    @reset vec.y.BCs = ()
    @reset vec.z.BCs = ()
    @reset vec.BCs = ()

    # Loop over boundary condition arguments to set boundary condition vectors
    for arg ∈ args

        # Set boundary index and retrieve corresponding name
        idx = boundary_index(model.boundary_info, arg.ID)
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
assign(scalar::ScalarField, model, args...) = begin

    # Retrieve user selected float type and boundaries
    float = _get_float(scalar.mesh)
    boundaries = scalar.mesh.boundaries

    # Assign tuples for boundary condition scalar
    @reset scalar.BCs = ()

    # Loop over boundary condition arguments to set boundary condition scalar
    for arg ∈ args

        # Set boundary index and retrieve corresponding name
        idx = boundary_index(model.boundary_info, arg.ID)
        bname = boundaries[idx].name
        println("Setting boundary $idx: ", bname)

        # Exception 1: value is a number
        if typeof(arg.value) <: Number
            BCs = (fixedValue(arg, idx, float(arg.value))) # doesn't work with tuples
            @reset scalar.BCs = (scalar.BCs..., BCs)

        # Exception 2: value is a named tuple (used in wall functions)
        elseif typeof(arg.value) <: NamedTuple
            BCs_vals = arg.value
            # for entry ∈ typeof(arg.value).parameters[1] # access names
            #     val = float(getproperty(arg.value, entry)) # type conversion
            #     BCs_vals = set(BCs_vals, PropertyLens{entry}(), val)
            # end
            println(arg)
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
        f = assign(f, $emodel, $eBCs...)
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
        f = assign(f, $emodel, $eBCs...)
        $emodel = @set $emodel.$eturb.$efield = f
    end
end

# Set schemes function definition with default set variables
set_schemes(;
    time=Steady,
    divergence=Linear, 
    laplacian=Linear, 
    gradient=Orthogonal) = begin
    
    # Tuple definition for scheme 
    (
        time=time,
        divergence=divergence,
        laplacian=laplacian,
        gradient=gradient
    )
end