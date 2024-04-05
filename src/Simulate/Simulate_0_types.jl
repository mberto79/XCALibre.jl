export Configuration

@kwdef struct Configuration{SC,SL,RT}
    schemes::SC
    solvers::SL
    runtime::RT
end
Adapt.@adapt_structure Configuration


# Simulation medium 
# struct Fluid{T} end 

# struct Incompressible end
# struct Compressible end

# struct Simulation{T,D,M,E}
#     type
#     domain
#     medium
#     energy
# end

